# app.py - AcadémIA API con TinyLlama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os

print("⏳ Cargando AcadémIA API (TinyLlama)...")

MODELO_NOMBRE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODELO_NOMBRE)
    model = AutoModelForCausalLM.from_pretrained(
        MODELO_NOMBRE,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✅ AcadémIA API cargada correctamente!")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None
    tokenizer = None

app = FastAPI(
    title="AcadémIA API",
    description="API de investigación académica con TinyLlama",
    version="1.0.0"
)

class Consulta(BaseModel):
    mensaje: str
    max_tokens: int = 500
    temperatura: float = 0.7

@app.post("/chat")
async def chat_endpoint(consulta: Consulta):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        inputs = tokenizer(consulta.mensaje, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=consulta.max_tokens,
            temperature=consulta.temperatura,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if respuesta.startswith(consulta.mensaje):
            respuesta = respuesta[len(consulta.mensaje):].strip()
        
        return {"respuesta": respuesta, "modelo": "TinyLlama"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "api": "AcadémIA API",
        "modelo": MODELO_NOMBRE,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "api": "AcadémIA API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

