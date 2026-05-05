import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==========================================
# 設定區
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "sft_data.json"
OUTPUT_DIR = "./sft_output"
FINAL_OUTPUT_DIR = f"{OUTPUT_DIR}/final"

def load_and_format_data(tokenizer):
    """
    讀取我們在 Step 2 準備的 sft_data.json，
    並套用 Qwen 專屬的 Chat Template。
    """
    print(f"📂 讀取訓練資料：{DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    # 轉換成 Hugging Face Dataset
    dataset = Dataset.from_list(raw_data)
    
    # 將 messages 轉換成模型看得懂的單一字串 (包含特殊的 special tokens)
    def apply_template(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], 
                tokenize=False, 
                add_generation_prompt=False
            )
        }
        
    formatted_dataset = dataset.map(apply_template)
    print(f"✅ 成功載入 {len(formatted_dataset)} 筆資料。")
    return formatted_dataset

def main():
    print("=" * 50)
    print("🚀 Lab 6: 開始 SFT 冷起動訓練")
    print("=" * 50)

    # 1. 準備 Tokenizer
    print("📦 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 準備訓練資料
    train_dataset = load_and_format_data(tokenizer)

    # 3. 設定 4-bit 量化 (節省 VRAM)
    print("📦 載入基礎模型 (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 準備模型以進行 k-bit 訓練 (開啟 Gradient Checkpointing 節省記憶體)
    base_model = prepare_model_for_kbit_training(base_model)

    # 4. 設定 LoRA 參數
    print("🔧 套用 LoRA Adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # 針對 Qwen / LLaMA 架構常見的 Attention 與 MLP 權重層
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. 設定訓練參數 (SFTConfig)
    print("⏳ 設定訓練參數...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",    # 指定剛剛 format 好的 text 欄位
        max_seq_length=1024,          # GSM8K 的長度通常不會超過 1024
        num_train_epochs=3,           # 跑 3 個 Epoch 確保模型記住格式
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,# 2 * 8 = 16 的等效 Batch Size
        learning_rate=2e-4,           # SFT 常用的學習率
        bf16=True,                    # 使用 bfloat16 加速並減少記憶體
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
    )

    # 6. 初始化 Trainer 並開始訓練
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    print("\n🔥 開始訓練 (請觀察 Loss 是否穩定下降)...")
    trainer.train()

    # 7. 儲存最終模型 (LoRA 權重)
    print(f"\n💾 儲存 LoRA 模型至 {FINAL_OUTPUT_DIR}...")
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
    print("✅ SFT 訓練完成！")

if __name__ == "__main__":
    main()