# SageMaker IAM and endpoint resources.

resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

locals {
  hf_container_image = "763104351884.dkr.ecr.${var.aws_region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
}

resource "aws_sagemaker_model" "llm" {
  name = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = local.hf_container_image
    environment = {
      HF_MODEL_ID = var.huggingface_model_id
      HF_TASK = "text-generation"
      HF_API_TOKEN = var.huggingface_api_token
      SM_NUM_GPUS = "1"
      MAX_INPUT_LENGTH = "2048"
      MAX_TOTAL_TOKENS = "4096"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "llm_config" {
  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name = "AllTraffic"
    model_name = aws_sagemaker_model.llm.name
    initial_instance_count = 1
    instance_type = var.sagemaker_instance_type
    initial_variant_weight = 1.0
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint" "llm_endpoint" {
  name = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.llm_config.name
}
