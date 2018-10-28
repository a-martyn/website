variable "access_key" {}
variable "secret_key" {}
variable "region" {
  default = "eu-west-2"
}
variable "aws_account_number" {}

variable "aliases" {
  default = ["www.alanmartyn.com", "alanmartyn.com"]
}

variable "acm_certificate_arn" {
  default = "arn:aws:acm:us-east-1:566664121427:certificate/4bdcff89-6df5-40ac-a44e-89630c0b4b7a"
}