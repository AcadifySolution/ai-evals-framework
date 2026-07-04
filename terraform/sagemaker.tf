# --- IAM Execution Role for SageMaker ---

resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AmazonSageMakerFullAccess policy
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Attach permissions to read/write from ECR (pull images) and S3 (models storage)
resource "aws_iam_role_policy_attachment" "sagemaker_s3_readonly" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}


# --- SageMaker Model with HuggingFace TGI Container ---

# Dynamic lookup for standard Hugging Face PyTorch TGI Inference ECR image
# The ECR registry account ID '763104351884' holds AWS deep learning containers.
locals {
  hf_container_image = "763104351884.dkr.ecr.${var.aws_region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
}

resource "aws_sagemaker_model" "llm" {
  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = local.hf_container_image

    environment = {
      HF_MODEL_ID           = var.huggingface_model_id
      HF_TASK               = "text-generation"
      HF_API_TOKEN          = var.huggingface_api_token
      SM_NUM_GPUS           = "1"
      MAX_INPUT_LENGTH      = "2048"
      MAX_TOTAL_TOKENS      = "4096"
      MAX_BATCH_PREFILL_TOKENS = "4096"
    }
  }
}


# --- SageMaker Endpoint Configuration ---

resource "aws_sagemaker_endpoint_configuration" "llm_config" {
  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.llm.name
    initial_instance_count = 1
    instance_type          = var.sagemaker_instance_type
    initial_variant_weight = 1.0
  }

  lifecycle {
    create_before_destroy = true
  }
}


# --- SageMaker Endpoint ---

resource "aws_sagemaker_endpoint" "llm_endpoint" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.llm_config.name

  tags = {
    Name = "${var.project_name}-sagemaker-endpoint"
  }
}
