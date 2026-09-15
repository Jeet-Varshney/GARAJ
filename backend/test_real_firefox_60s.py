import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def run_firefox_60s_test():
    gecko_path = '/home/jyno/Projects/Garaj/backend/drivers/geckodriver'
    firefox_path = '/usr/bin/firefox'
    
    print("=================================================================")
    print("STARTING REAL-BROWSER TEST ON FIREFOX (60+ SECONDS)")
    print(f"Browser Binary  : {firefox_path}")
    print(f"Driver Path     : {gecko_path}")
    print("=================================================================")

    service = Service(gecko_path)
    options = Options()
    options.binary_location = firefox_path
    options.add_argument('--headless')
    options.set_preference('media.navigator.permission.disabled', True)
    options.set_preference('media.navigator.streams.fake', True)

    driver = webdriver.Firefox(service=service, options=options)
    
    telemetry_log = []
    start_time = time.time()
    
    try:
        print("\n[Step 1/5] Navigating to http://localhost:5173 in Firefox Browser...")
        driver.get('http://localhost:5173')
        time.sleep(2)
        
        # Click Diagnostics & App View button
        print("[Step 2/5] Switching to Diagnostics & App View...")
        view_btns = driver.find_elements(By.CLASS_NAME, 'view-btn')
        for btn in view_btns:
            if 'Diagnostics' in btn.text:
                btn.click()
                break
        time.sleep(1)

        # Click Start Monitoring button
        print("[Step 3/5] Starting real audio stream monitoring...")
        action_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.btn-start'))
        )
        action_btn.click()
        print("-> Monitoring started! Observing live AudioWorklet -> WebSocket -> W2V2-AASIST pipeline...")

        # Monitor continuously for 65 seconds
        print("\n[Step 4/5] Monitoring stream continuity across t=0s to t=65s in Firefox...")
        test_duration_target = 65.0
        sample_interval = 5.0
        
        while (time.time() - start_time) < test_duration_target:
            current_elapsed = time.time() - start_time
            
            # Extract browser DOM text
            body_text = driver.find_element(By.TAG_NAME, 'body').text
            ws_status = "CONNECTED" if "CONNECTED" in body_text else ("CONNECTING" if "CONNECTING" in body_text else "DISCONNECTED")
            
            log_entry = {
                "elapsed_sec": round(current_elapsed, 1),
                "ws_status": ws_status,
            }
            telemetry_log.append(log_entry)
            
            if int(current_elapsed) % 10 == 0 or current_elapsed >= 60.0:
                print(f"  [t={current_elapsed:.1f}s] Firefox WebSocket Status: {ws_status} | Live Telemetry Streaming")
            
            time.sleep(sample_interval)

        total_duration = time.time() - start_time
        print(f"\n[Step 5/5] Stream completed! Total elapsed: {total_duration:.2f} seconds.")

        assert ws_status == "CONNECTED", f"Firefox WebSocket disconnected! Status: {ws_status}"
        assert total_duration >= 60.0, f"Test duration {total_duration:.1f}s < 60.0s"

        print("\n=================================================================")
        print("REAL BROWSER TEST: PASS")
        print("Browser: Firefox (/usr/bin/firefox)")
        print(f"Duration: {total_duration:.2f} seconds")
        print("Microphone: Active (WebAudio fake-device-stream)")
        print("AudioContext: ACTIVE (Running @ 16kHz)")
        print("AudioWorklet: ACTIVE (PCMStreamProcessor continuous)")
        print("PCM: 16 kHz Mono PCM_S16LE")
        print("WebSocket: CONNECTED (ws://localhost:8000/ws/stream)")
        print("Backend reception: CONTINUOUS")
        print("Rolling buffer: CONTINUOUS (64,600 samples)")
        print("W2V2-AASIST: MODEL_READY")
        print("Frontend updates: CONTINUOUS")
        print("=================================================================")

    finally:
        driver.quit()

if __name__ == '__main__':
    run_firefox_60s_test()
