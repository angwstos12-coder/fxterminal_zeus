import asyncio
import os
import time
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from engine import SmartChartEngine, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

WORKSPACE_CACHE: dict[str, dict] = {}
ACTIVE_CLIENTS: list[tuple[WebSocket, str]] = []

# Μνήμη tracking: { "USDCAD": {"prediction": "...", "confidence": 85.0, "timestamp": 12345678} }
LAST_SENT_ALERTS: dict[str, dict] = {}
COOLDOWN_SECONDS = 3600  # 1 ώρα minimum cooldown, εκτός αν υπάρξει μεγάλη αλλαγή ποσοστού/κατεύθυνσης
CONFIDENCE_SHIFT_THRESHOLD = 5.0  # Αν το ποσοστό αλλάξει κατά >= 5%, στέλνει update

def send_telegram_notification(text: str):
    """Βοηθητική συνάρτηση αποστολής ειδοποίησης στο Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🤖 *AI INSTITUTIONAL BOT UPDATE*\n\n{text}",
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[Telegram Background Error]: {e}")

async def auto_refresh_and_alert_loop():
    global ACTIVE_CLIENTS, LAST_SENT_ALERTS
    scan_pairs = ["USDCAD", "EURUSD", "GBPUSD", "GBPJPY", "XAUUSD"]
    
    while True:
        try:
            active_pairs = list(set([pair for _, pair in ACTIVE_CLIENTS] + scan_pairs))
            current_time = time.time()
            
            for pair_key in active_pairs:
                eng = SmartChartEngine(pair_key)
                data = await asyncio.to_thread(eng.build_full_workspace)
                WORKSPACE_CACHE[pair_key] = data

                nn = data.get("neural_prediction", {})
                confidence = nn.get("confidence_pct", 0)
                prediction = nn.get("prediction", "")
                action = nn.get("trade_action", "")

                # Αν η εμπιστοσύνη είναι >= 75% και δεν είναι WAIT
                if confidence >= 75 and "WAIT" not in action:
                    last_record = LAST_SENT_ALERTS.get(pair_key, {})
                    last_pred = last_record.get("prediction", "")
                    last_conf = last_record.get("confidence", 0.0)
                    last_time = last_record.get("timestamp", 0)
                    
                    # Κριτήρια ενημέρωσης:
                    is_prediction_changed = (last_pred != prediction)
                    is_confidence_shifted = (abs(confidence - last_conf) >= CONFIDENCE_SHIFT_THRESHOLD)
                    is_cooldown_expired = (current_time - last_time > COOLDOWN_SECONDS)

                    # Στέλνει αν άλλαξε κατεύθυνση, ή αν άλλαξε σημαντικά το ποσοστό, ή αν πέρασε η ώρα
                    if is_prediction_changed or (is_confidence_shifted and is_cooldown_expired) or not last_record:
                        
                        update_reason = "🔄 Αλλαγή Κατεύθυνσης" if is_prediction_changed else ("📈 Δυναμική Μεταβολή Ποσοστού AI" if is_confidence_shifted else "⏰ Περιοδικό Update")
                        
                        alert_msg = (
                            f"🔥 *High-Confidence AI Signal* 🔥\n"
                            f"_Reason: {update_reason}_\n\n"
                            f"▫️ *Asset:* `{pair_key}`\n"
                            f"▫️ *Prediction:* `{prediction}`\n"
                            f"▫️ *Confidence:* `{confidence}%` *(προηγ. {last_conf}%)*\n"
                            f"▫️ *Multi-TF Status:* `{nn.get('time_horizon', 'N/A')}`\n"
                            f"▫️ *Action:* `{action}`\n"
                            f"▫️ *Win-Rate (Forward Test):* `{data.get('forward_testing', {}).get('win_rate_pct', 0)}%`"
                        )
                        send_telegram_notification(alert_msg)
                        
                        # Ενημέρωση μνήμης
                        LAST_SENT_ALERTS[pair_key] = {
                            "prediction": prediction,
                            "confidence": confidence,
                            "timestamp": current_time
                        }

                # Ενημέρωση συνδεδεμένων WebSockets clients στον browser
                for ws, client_pair in list(ACTIVE_CLIENTS):
                    if client_pair == pair_key:
                        try:
                            await ws.send_json(data)
                        except Exception:
                            if (ws, client_pair) in ACTIVE_CLIENTS:
                                ACTIVE_CLIENTS.remove((ws, client_pair))
                                
        except Exception as e:
            print(f"[Background Loop Error]: {e}")
            
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_engine = SmartChartEngine("USDCAD")
    WORKSPACE_CACHE["USDCAD"] = await asyncio.to_thread(init_engine.build_full_workspace)
    task = asyncio.create_task(auto_refresh_and_alert_loop())
    yield
    task.cancel()

app = FastAPI(title="Smart Chart Intelligence API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/workspace/{pair}")
async def get_workspace(pair: str):
    pair_key = pair.upper().replace("/", "")
    if pair_key not in WORKSPACE_CACHE:
        temp_engine = SmartChartEngine(pair_key)
        WORKSPACE_CACHE[pair_key] = await asyncio.to_thread(temp_engine.build_full_workspace)
    return WORKSPACE_CACHE[pair_key]

@app.websocket("/ws/workspace/{pair}")
async def websocket_endpoint(websocket: WebSocket, pair: str):
    global ACTIVE_CLIENTS
    await websocket.accept()
    pair_key = pair.upper().replace("/", "")
    ACTIVE_CLIENTS.append((websocket, pair_key))
    try:
        if pair_key not in WORKSPACE_CACHE:
            temp_engine = SmartChartEngine(pair_key)
            WORKSPACE_CACHE[pair_key] = await asyncio.to_thread(temp_engine.build_full_workspace)
        await websocket.send_json(WORKSPACE_CACHE[pair_key])
        
        while True:
            data = await websocket.receive_text()
            if data.startswith("SWITCH:"):
                new_p = data.split(":")[1].upper().replace("/", "")
                ACTIVE_CLIENTS = [(w, cp) for w, cp in ACTIVE_CLIENTS if w != websocket]
                ACTIVE_CLIENTS.append((websocket, new_p))
                if new_p not in WORKSPACE_CACHE:
                    temp_engine = SmartChartEngine(new_p)
                    WORKSPACE_CACHE[new_p] = await asyncio.to_thread(temp_engine.build_full_workspace)
                await websocket.send_json(WORKSPACE_CACHE[new_p])
    except WebSocketDisconnect:
        ACTIVE_CLIENTS = [(w, cp) for w, cp in ACTIVE_CLIENTS if w != websocket]

@app.get("/")
async def serve_dashboard():
    return FileResponse("index.html")