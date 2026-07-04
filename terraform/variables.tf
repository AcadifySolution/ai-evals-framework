variable "aws_region" {
  description = "Target AWS Region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Target execution environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Name identifier for resource tagging"
  type        = string
  default     = "ai-evals-framework"
}

variable "vpc_cidr" {
  description = "IP block configuration for isolated VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS DB Instance type for eval logs storage"
  type        = string
  default     = "db.t4g.micro"
}

variable "db_name" {
  description = "PostgreSQL initial database name"
  type        = string
  default     = "aievalsdb"
}

variable "db_username" {
  description = "PostgreSQL administrator login name"
  type        = string
  default     = "evaladmin"
}

variable "db_password" {
  description = "PostgreSQL administrator password (should be set via TF_VAR_db_password)"
  type        = string
  sensitive   = true
  default     = "SuperSecurePassword123!"
}

variable "sagemaker_instance_type" {
  description = "GPU computing power type for hosting the LLM under test"
  type        = string
  default     = "ml.g5.2xlarge" # Default single GPU (A10G) suitable for 7B-8B parameter models
}

variable "huggingface_model_id" {
  description = "The target model repository identifier from HF Hub"
  type        = string
  default     = "meta-llama/Meta-Llama-3-8B-Instruct"
}

variable "huggingface_api_token" {
  description = "Hugging Face Access Token for gated models (e.g. Llama 3)"
  type        = string
  sensitive   = true
  default     = "hf_placeholder"
}
