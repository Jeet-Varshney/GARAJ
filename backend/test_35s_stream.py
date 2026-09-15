import asyncio
import websockets
import json
import struct
import time

async def test_35s_decoupled_stream():
    uri = 'ws://localhost:8000/ws/stream'
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        print("Connected successfully. Starting 35-second decoupled live streaming (350 chunks @ 100ms pace)...")
        
        chunk_pcm = b'\x00' * 3200
        responses_received = 0
        statuses = []
        stop_receiving = False
        start_wall_time = time.time()

        # Receiver Task
        async def receiver_loop():
            nonlocal responses_received, stop_receiving
            while not stop_receiving or responses_received < 350:
                try:
                    resp_bytes = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(resp_bytes)
                    responses_received += 1
                    det_status = data.get("detection", {}).get("status")
                    det_class = data.get("detection", {}).get("predicted_class")
                    seq_id = data.get("seq_id", -1)
                    statuses.append((seq_id, det_status, det_class))
                    
                    if responses_received % 50 == 0 or responses_received == 350:
                        elapsed = time.time() - start_wall_time
                        print(f"[Received Response #{responses_received}/350 | Elapsed {elapsed:.1f}s] seq_id={seq_id} | status={data.get('status')} | detection_status={det_status} | predicted_class={det_class}")
                except asyncio.TimeoutError:
                    if stop_receiving and responses_received >= 350:
                        break
                    print(f"Waiting for remaining responses... ({responses_received}/350 received)")

        recv_task = asyncio.create_task(receiver_loop())

        # Sender Loop (simulates live AudioWorklet / mic sending 1 chunk every 100ms)
        for seq_id in range(350): # 35 seconds of continuous 100ms chunks
            client_ts_ms = time.time() * 1000.0
            header = struct.pack('<Id', seq_id, client_ts_ms)
            payload = header + chunk_pcm
            
            await ws.send(payload)
            await asyncio.sleep(0.1)

        send_elapsed = time.time() - start_wall_time
        print(f"\n---> Sent all 350 chunks in {send_elapsed:.2f}s (Paced mic audio stream completed). Waiting for remaining responses...")
        stop_receiving = True

        await asyncio.wait_for(recv_task, timeout=15.0)

        total_elapsed = time.time() - start_wall_time
        print(f"\n=======================================================")
        print(f"Test Finished! Total elapsed: {total_elapsed:.2f}s")
        print(f"Total responses received: {responses_received}/350")
        print(f"Final response: {statuses[-1]}")
        print(f"=======================================================")

        assert responses_received == 350, f"Expected 350 responses, got {responses_received}"
        print("PASS: 35-Second Decoupled Live Audio Streaming Verified!")

if __name__ == "__main__":
    asyncio.run(test_35s_decoupled_stream())
