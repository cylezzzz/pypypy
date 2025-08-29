import subprocess
import time
import requests
import sys

def test_server_startup():
    """Testet ob der Server startet"""
    
    print("🚀 === SERVER-TEST ===")
    
    try:
        # Starte Server im Hintergrund
        print("🔄 Starte Server...")
        process = subprocess.Popen([
            sys.executable, "start.py", "--debug"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Warte kurz
        time.sleep(10)
        
        # Teste ob Server antwortet
        try:
            response = requests.get("http://localhost:3000", timeout=5)
            if response.status_code == 200:
                print("✅ Server läuft erfolgreich!")
                print("🌐 Öffne: http://localhost:3000")
                return True
            else:
                print(f"⚠️ Server antwortet mit Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Server nicht erreichbar: {e}")
            return False
            
        finally:
            # Server beenden
            process.terminate()
            
    except Exception as e:
        print(f"❌ Server-Start Fehler: {e}")
        return False

if __name__ == "__main__":
    test_server_startup()