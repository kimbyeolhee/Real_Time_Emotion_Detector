import argparse
from omegaconf import OmegaConf

from transformers import ViTForImageClassification, ViTImageProcessor

def main(config, huggingface_api_key):
    model = ViTForImageClassification.from_pretrained(config.training_args.output_dir)
    processor = ViTImageProcessor.from_pretrained(config.training_args.output_dir)

    model.push_to_hub("emotion-classifier-vit", use_temp_dir=True, use_auth_token=huggingface_api_key)
    processor.push_to_hub("emotion-classifier-vit", use_temp_dir=True, use_auth_token=huggingface_api_key)

    print("Model and processor uploaded to Hugging Face successfully")
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config", help="config file path")
    parser.add_argument("--huggingface_api_key", "-k", type=str, required=True, help="Hugging Face API key")
    args, _ = parser.parse_known_args()

    config = OmegaConf.load(f"configs/{args.config}.yaml")
    main(config, args.huggingface_api_key)