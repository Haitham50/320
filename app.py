import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Configuration
MODEL_NAME = "Qwen/Qwen-0.5B-Chat"
SYSTEM_PROMPT = "أنت مساعد ذكاء اصطناعي مفيد وودود، تجيب على الأسئلة باللغة العربية بطلاقة."

# 2. Model Loading with Quantization
try:
    # Use 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Successfully loaded model {MODEL_NAME} with 4-bit quantization.")

except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    # Fallback to a very small, non-quantized model if Qwen fails
    MODEL_NAME = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    SYSTEM_PROMPT = "أنت مساعد ذكاء اصطناعي بسيط. لا يمكنني معالجة اللغة العربية بشكل جيد بسبب قيود الموارد."
    print(f"Fallback to {MODEL_NAME} due to resource constraints.")


# 3. Chat Function
def chat_with_model(message, history):
    # Format chat history for Qwen model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Tokenize and generate response
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response with streaming
    streamer = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.02,
        pad_token_id=tokenizer.eos_token_id,
        streamer=None # Gradio handles streaming internally
    )

    # Decode and yield the response
    full_response = ""
    for new_token in streamer:
        if new_token.item() != tokenizer.eos_token_id:
            full_response += tokenizer.decode(new_token, skip_special_tokens=True)
            yield full_response

# 4. Gradio Interface
gr.ChatInterface(
    chat_with_model,
    title=f"وكيل الذكاء الاصطناعي (النموذج: {MODEL_NAME})",
    description="مساعد ذكاء اصطناعي يدعم اللغة العربية. يرجى ملاحظة أن الأداء قد يكون محدودًا بسبب قيود الموارد في المساحة المجانية.",
    theme="soft",
    submit_btn="إرسال",
    retry_btn="إعادة المحاولة",
    undo_btn="تراجع",
    clear_btn="مسح المحادثة",
).queue().launch()
