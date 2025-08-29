#!/usr/bin/env python3
"""
AndioMediaStudio - Universal Local AI Media Studio
Enhanced startup script with better configuration and monitoring
"""

import uvicorn
import os
import sys
import signal
import logging
from pathlib import Path
import yaml
from typing import Dict, Any
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AndioMediaStudio")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "server" / "config" / "settings.yaml"

def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml with fallbacks"""
    default_config = {
        'port': 3000,
        'host': '0.0.0.0',
        'reload': False,
        'log_level': 'info',
        'workers': 1,
        'mode': 'creative',
        'max_parallel_workers': 1,
        'prefer_gpu': True
    }
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                return {**default_config, **config}
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")
    return default_config

def setup_directories():
    """Ensure all required directories exist"""
    dirs_to_create = [
        BASE_DIR / "models",
        BASE_DIR / "outputs" / "images",
        BASE_DIR / "outputs" / "videos",
        BASE_DIR / "workspace",
        BASE_DIR / "server" / "temp"
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ready: {directory}")

def check_dependencies():
    """Check if critical dependencies are available"""
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠ No CUDA GPU detected, using CPU")
    except ImportError:
        logger.error("✗ PyTorch not found - install with: pip install torch")
    try:
        import diffusers
        logger.info(f"✓ Diffusers: {diffusers.__version__}")
    except ImportError:
        logger.warning("⚠ Diffusers not found - some features may not work")

def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def print_startup_banner():
    """Print startup information"""
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                  🎨 AndioMediaStudio v0.5               ║
║              Universal Local AI Media Studio             ║
╚══════════════════════════════════════════════════════════╝

🚀 Features:
   • Text → Image (Stable Diffusion, FLUX, etc.)
   • Image → Image & Inpainting
   • Pose Transfer & ControlNet
   • Text → Video & Image → Video
   • Lip-Sync & Talking Head
   • Auto Masking (SAM/MediaPipe)
   • No Content Filters - Full Creative Control

📂 Project Structure:
   • Models: {BASE_DIR / 'models'}
   • Outputs: {BASE_DIR / 'outputs'}
   • Workspace: {BASE_DIR / 'workspace'}
"""
    print(banner)

async def run_server(config: Dict[str, Any]):
    """Run the FastAPI server with enhanced configuration"""

    # FIX: Standard-App jetzt korrekt auf server.app:app
    app_target = os.environ.get("ANDIO_APP", "server.app:app")

    # Ensure project root is importable
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Env
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    server_config = uvicorn.Config(
        app_target,
        host=config.get('host', '0.0.0.0'),
        port=int(config.get('port', 3000)),
        reload=bool(config.get('reload', False)),
        log_level=str(config.get('log_level', 'info')),
        workers=int(config.get('workers', 1)),
        access_log=True,
        use_colors=True
    )
    server = uvicorn.Server(server_config)

    logger.info(f"🌐 Starting server on http://{config['host']}:{config['port']}")
    logger.info(f"🖥  Local access: http://localhost:{config['port']}")
    logger.info(f"🧭 App target: {app_target}")

    if config['host'] == '0.0.0.0':
        import socket
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            logger.info(f"🌍 Network access: http://{local_ip}:{config['port']}")
        except Exception:
            pass

    await server.serve()

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print_startup_banner()
        config = load_config()
        logger.info(f"📋 Loaded configuration: {CONFIG_FILE}")
        setup_directories()
        check_dependencies()

        # Override with environment variables
        port = int(os.environ.get('PORT', config.get('port', 3000)))
        host = os.environ.get('HOST', config.get('host', '0.0.0.0'))
        config.update({'port': port, 'host': host})

        asyncio.run(run_server(config))

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
