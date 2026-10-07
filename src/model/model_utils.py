# model_utils.py - enhanced version

from model.openai_compat import DEFAULT_MAX_TOKENS
from personas import _build_chosen_personas, _build_enhanced_personas

# The generation budget every fallback resolves to. It used to be 512 here and
# a hardcoded 4096 inside OpenAICompatChatWrapper.complete, which overwrote it,
# so 4096 is what every vLLM run has actually been using. Keeping that as the
# default makes the flag real without moving any existing result.
DEFAULT_MAX_NEW_TOKENS = DEFAULT_MAX_TOKENS

model_dirs = {
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct'
}

AZURE_OPENAI_MODELS = {
    'gpt-4o', 'gpt-4o-mini',
    'o1', 'o3-mini', 'o3', 'o4-mini',
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
    'Kimi-K2.5', 'DeepSeek-V3.2', 'claude-sonnet-4-6', 'ministral-3b', 'Llama-3.3-70B-Instruct'
}

# Closed-source models (accessed via OpenAI-compatible API)
OPENAI_COMPAT_MODELS = {
    'gpt-5-mini', 'gpt-4.1-mini', 'gemini-2.5-flash',
}


def _split_csv(s: str):
    return [x.strip() for x in (s or '').split(',') if x.strip()]


def engine(messages, agent, num_agents=1, stop_sequences=None, persona_configs=None):
    """
    Unified handler: messages is always [{"role": "user", "content": "..."}, ...]
    persona_configs: optional list of dict, per-agent personalized configuration
                     format: [{"temperature": 0.3, "top_p": 0.85}, ...]
    """
    def _run_one(current_agent, msg, config=None):
        """Run one agent on one message with optional per-agent config."""
        # Get generation parameters: prefer config, fall back to agent defaults
        temperature = config.get('temperature', getattr(current_agent, 'temperature', 1.0)) if config else getattr(current_agent, 'temperature', 1.0)
        top_p = config.get('top_p', getattr(current_agent, 'top_p', 0.9)) if config else getattr(current_agent, 'top_p', 0.9)
        max_new_tokens = config.get('max_new_tokens', getattr(current_agent, 'max_new_tokens', DEFAULT_MAX_NEW_TOKENS)) if config else getattr(current_agent, 'max_new_tokens', DEFAULT_MAX_NEW_TOKENS)

        # API-like chat agents (Azure OpenAI or local OpenAI-compatible servers like vLLM)
        if getattr(current_agent, 'kind', None) in {'azure_openai', 'openai_compat'}:
            return current_agent.complete(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg['content']},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                #logprobs=True
            )

        # Local HF agent
        prompts = [msg['content']]
        inputs = current_agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(current_agent.huggingface_model.device)
        attention_mask = inputs['attention_mask'].to(current_agent.huggingface_model.device)
        outputs = current_agent.huggingface_model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=current_agent.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            return_legacy_cache=True
        )
        sequence = outputs.sequences[0]
        gen_only = sequence[len(input_ids[0]):]
        decoded = current_agent.tokenizer.decode(gen_only, skip_special_tokens=True)
        return decoded

    # Handle persona_configs
    if persona_configs is None:
        persona_configs = [None] * len(messages)
    elif len(persona_configs) < len(messages):
        # Cycle-extend to match message count
        persona_configs = [persona_configs[i % len(persona_configs)] for i in range(len(messages))]

    # Support heterogeneous agents
    if isinstance(agent, (list, tuple)):
        responses = []
        for i, msg in enumerate(messages):
            current_agent = agent[i % len(agent)]
            config = persona_configs[i] if persona_configs else None
            responses.append(_run_one(current_agent, msg, config))
        return responses

    # Homogeneous mode (single agent)
    if getattr(agent, 'kind', None) in {'azure_openai', 'openai_compat'}:
        return [_run_one(agent, msg, persona_configs[i] if persona_configs else None) 
                for i, msg in enumerate(messages)]
    
    # For HF models: if configs differ across agents, process one by one
    if persona_configs and any(c is not None for c in persona_configs):
        # With personalized configs, process individually
        return [_run_one(agent, msg, persona_configs[i]) for i, msg in enumerate(messages)]
    
    # No personalized configs: batch processing
    prompts = [msg['content'] for msg in messages]
    inputs = agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(agent.huggingface_model.device)
    attention_mask = inputs['attention_mask'].to(agent.huggingface_model.device)

    outputs = agent.huggingface_model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=agent.tokenizer.eos_token_id,
        max_new_tokens=getattr(agent, 'max_new_tokens', DEFAULT_MAX_NEW_TOKENS),
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        temperature=getattr(agent, 'temperature', 1.0),
        top_p=getattr(agent, 'top_p', 0.9),
        num_return_sequences=1,
        return_legacy_cache=True
    )

    generated_sequences = outputs.sequences
    responses = []
    for input_id, sequence in zip(input_ids, generated_sequences):
        gen_only = sequence[len(input_id):]
        decoded = agent.tokenizer.decode(gen_only, skip_special_tokens=True)
        responses.append(decoded)

    return responses


def get_agents(args, peft_path=None):
    agent_model_keys = []
    if getattr(args, 'agent_models', ''):
        agent_model_keys = [m.strip() for m in args.agent_models.split(',') if m.strip()]
    if not agent_model_keys:
        agent_model_keys = [args.model]
    args.agent_model_keys = agent_model_keys

    def _make_agent(model_key: str, vllm_base_url=None):
        import os
        
        # Closed-source models (via OpenAI-compatible API)
        # Priority check: if OPENAI_BASE_URL is set and model is in OPENAI_COMPAT_MODELS
        openai_base_url = getattr(args, 'openai_base_url', '') or os.getenv('OPENAI_BASE_URL', '')
        if model_key in OPENAI_COMPAT_MODELS or (openai_base_url and model_key in AZURE_OPENAI_MODELS):
            from model.openai_compat import OpenAICompatChatWrapper
            api_key = getattr(args, 'openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')
            base_url = openai_base_url or 'https://api.openai.com/v1'
            if not api_key:
                raise ValueError(f"OpenAI API key not found for model {model_key}. Set --openai_api_key or OPENAI_API_KEY env var.")
            return OpenAICompatChatWrapper(
                base_url=base_url,
                model_name=model_key,
                api_key=api_key,
            )

        if model_key in AZURE_OPENAI_MODELS:
            from model.azure_openai import AzureOpenAIWrapper
            api_key = getattr(args, 'azure_api_key_env', 'API_KEY')
            if not getattr(args, 'azure_endpoint', ''):
                raise ValueError("azure_endpoint is empty.")
            if not api_key:
                print(args)
                raise ValueError(f"Azure OpenAI API key not found.")
            return AzureOpenAIWrapper(
                model_name=model_key,
                azure_endpoint=args.azure_endpoint,
                api_key=api_key,
                api_version=getattr(args, 'azure_api_version', '2025-04-01-preview'),
            )

        if getattr(args, 'use_vllm', False):
            from model.openai_compat import OpenAICompatChatWrapper
            base_url = (vllm_base_url or getattr(args, 'vllm_base_url', 'http://127.0.0.1:8000/v1')).rstrip('/')
            return OpenAICompatChatWrapper(
                base_url=base_url,
                model_name=model_key,
                api_key=getattr(args, 'vllm_api_key', 'EMPTY'),
            )

        if model_key in ['llama3.1-8b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
            from model.llama import LlamaWrapper
            return LlamaWrapper(args, model_dirs[model_key], 
                              memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, 
                              lora_adapter_path=peft_path, llama_version=3)
        elif model_key in ['qwen2.5-7b', 'qwen2.5-32b']:
            from model.qwen import QwenWrapper
            return QwenWrapper(args, model_dirs[model_key], 
                             memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, 
                             lora_adapter_path=peft_path)
        else:
            raise ValueError(f"invalid model key: {model_key}")

    # Build enhanced personas
    if getattr(args, 'chosen_agents', True):
        personas = _build_chosen_personas(args)
    else:
        personas = _build_enhanced_personas(args)

    vllm_urls = _split_csv(getattr(args, 'vllm_base_urls', ''))
    if not vllm_urls:
        vllm_urls = [getattr(args, 'vllm_base_url', 'http://127.0.0.1:8001/v1')]

    if not getattr(args, 'agent_models', ''):
        print(vllm_urls)
        agent = _make_agent(args.model, vllm_base_url=vllm_urls[0])
        agent.max_new_tokens = getattr(args, 'max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        agent.temperature = getattr(args, 'temperature', 0)
        agent.top_p = getattr(args, 'top_p', 0.9)

        if hasattr(agent, 'tokenizer') and hasattr(agent, 'huggingface_model'):
            if agent.tokenizer.pad_token is None:
                agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                agent.huggingface_model.resize_token_embeddings(len(agent.tokenizer))

        return agent, personas

    agents = []
    for idx, mk in enumerate(agent_model_keys):
        a = _make_agent(mk, vllm_base_url=vllm_urls[idx % len(vllm_urls)])
        a.max_new_tokens = getattr(args, 'max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        a.temperature = getattr(args, 'temperature', 0)
        a.top_p = getattr(args, 'top_p', 0.9)
        if hasattr(a, 'tokenizer') and hasattr(a, 'huggingface_model'):
            if a.tokenizer.pad_token is None:
                a.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                a.huggingface_model.resize_token_embeddings(len(a.tokenizer))
        agents.append(a)

    return agents, personas
