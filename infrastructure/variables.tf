variable "aws_access_key" {}
variable "aws_secret_key" {}
variable "aws_region" {
  default = "eu-west-2"
}
variable "aws_account_number" {}

variable "aliases" {
  default = ["www.alanmartyn.com", "alanmartyn.com"]
}

variable "acm_certificate_arn" {}

variable "env" {
  description = "e.g. develop, live"
}