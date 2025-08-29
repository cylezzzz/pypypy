import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    AnimateDiffPipeline
)

class PipelineFactory:
    def __init__(self):
        self.loaded_pipelines = {}
        
    def get_pipeline(self, model_name, task_type="txt2img"):
        """Lade Pipeline für deine spezifischen Modelle"""
        
        # Definiere deine Modell-Mappings
        model_configs = {
            "realistic_vision": {
                "path": r"X:\pypygennew\models\image\Realistic_Vision_V6.0_B1_noVAE",
                "pipeline_class": StableDiffusionPipeline,
                "type": "sd15"
            },
            "sdxl_base": {
                "path": r"X:\pypygennew\models\image\stable-diffusion-xl-base-1.0", 
                "pipeline_class": StableDiffusionXLPipeline,
                "type": "sdxl"
            },
            "sdxl_refiner": {
                "path": r"X:\pypygennew\models\image\stable-diffusion-xl-refiner-1.0",
                "pipeline_class": StableDiffusionXLPipeline, 
                "type": "sdxl_refiner"
            },
            "svd_img2vid": {
                "path": r"X:\pypygennew\models\video\stable-video-diffusion-img2vid",
                "pipeline_class": StableVideoDiffusionPipeline,
                "type": "img2vid"
            }
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Modell nicht gefunden: {model_name}")
        
        # Lade Pipeline falls nicht bereits geladen
        if model_name not in self.loaded_pipelines:
            config = model_configs[model_name]
            
            try:
                print(f"🔄 Lade {model_name}...")
                
                pipeline = config["pipeline_class"].from_pretrained(
                    config["path"],
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=True,
                    variant="fp16" if "xl" in model_name else None
                )
                
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                    print(f"✅ {model_name} auf GPU geladen")
                else:
                    print(f"✅ {model_name} auf CPU geladen")
                
                self.loaded_pipelines[model_name] = pipeline
                
            except Exception as e:
                print(f"❌ Fehler beim Laden von {model_name}: {e}")
                return None
        
        return self.loaded_pipelines[model_name]