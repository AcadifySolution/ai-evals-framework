output "vpc_id" {
  description = "The ID of the generated VPC network"
  value       = aws_vpc.main.id
}

output "db_connection_endpoint" {
  description = "The hostname endpoint for the logging PostgreSQL instance"
  value       = aws_db_instance.evals_db.endpoint
}

output "sagemaker_endpoint_name" {
  description = "The string name of the deployed LLM SageMaker endpoint"
  value       = aws_sagemaker_endpoint.llm_endpoint.name
}

output "sagemaker_endpoint_arn" {
  description = "The Amazon Resource Name of the deployed SageMaker endpoint"
  value       = aws_sagemaker_endpoint.llm_endpoint.arn
}
