variable "aws_region" {
  description = "Target AWS Region"
  type = string
  default = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type = string
  default = "dev"
  validation {
    condition = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Resource name prefix"
  type = string
  default = "ai-evals-framework"
}

variable "vpc_cidr" {
  description = "VPC CIDR"
  type = string
  default = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type = string
  default = "db.t4g.micro"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type = string
  default = "aievalsdb"
}

variable "db_username" {
  description = "PostgreSQL application username"
  type = string
  default = "evalapp"
}

variable "db_password" {
  description = "PostgreSQL password supplied via TF_VAR_db_password or an external secret manager."
  type = string
  sensitive = true
  default = null
}

variable "sagemaker_instance_type" {
  description = "SageMaker GPU instance type"
  type = string
  default = "ml.g5.2xlarge"
}

variable "huggingface_model_id" {
  description = "Hugging Face model identifier"
  type = string
  default = "meta-llama/Meta-Llama-3-8B-Instruct"
}

variable "huggingface_api_token" {
  description = "Hugging Face token supplied at deployment time."
  type = string
  sensitive = true
  default = null
}
