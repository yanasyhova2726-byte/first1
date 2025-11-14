from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import threading, json, os, torch

#Новое
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")
#Конец нового

app = Flask(__name__)

model_name = "gpt2"          #os.getenv("MODEL_NAME", "tiiuae/falcon-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if device == "cuda" else torch.float32)
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, max_length=150) 

lock = threading.Lock()
MY_NAME, BOT_NAME = "Brain", "Yana"
MAX_TURNS = 20

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hello")
def hello():
    return "Здравствуйте!"

@app.route("/healthz")
def healthz():
    return jsonify(status="ok")

@app.route("/chat", methods=["POST"])
def chat():
    if request.is_json:
        p = request.get_json(force=True) or {}
        user_input = (p.get("user_input") or "").strip()
        dialog = p.get("dialog")
    else:
        user_input = (request.form.get("user_input") or "").strip()
        try:
            dialog = json.loads(request.form.get("dialog", "[]"))
        except Exception:
            dialog = None

    if not user_input:
        return jsonify(error="Пустое поле user_input"), 400
    if not isinstance(dialog, list):
        dialog = []
    if len(dialog) > MAX_TURNS:
        dialog = dialog[-MAX_TURNS:]

    dialog.append(f"{MY_NAME}: {user_input}")
    prompt = "\n".join(dialog) + f"\n{BOT_NAME}:"

    with lock:
            out = llm(prompt, max_new_tokens=200, do_sample=True, top_k=10,
                      temperature=0.7, num_return_sequences=1, return_full_text=False,
                      pad_token_id=pad_id)
            resp = None
            
            try:
                if isinstance(out, list) and len(out) > 0 and "generated text" in out [0] :
                    resp = out[0] ["generated_text"]  #Тут исправила добавила _
                else:
                    print(f"Несоответствие формата вывода pipeline: {out}")
                    return jsonify(error = "Модель вернула неожиданный формат"), 500
            except Exception as e:
                print(f"Ошибка при попытке извлечения сгенерированного текста: {e}")
                return jsonify(error = f"Ошибка при попытке извлечения сгенерированного текста: {e}"), 500
                        
    dialog.append(f"{BOT_NAME}: {resp}")
    return jsonify(response=resp, dialog=dialog)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
