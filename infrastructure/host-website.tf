resource "aws_s3_bucket" "host_website" {
    bucket = "host-website-${var.env}"
    website {
        index_document = "index.html"
        error_document = "404.html"

        routing_rules = <<EOF
        [{
            "Condition": {
                "HttpErrorCodeReturnedEquals": "400"
            },
            "Redirect": {
                "ReplaceKeyWith": "404.html"
            }
        }]
    EOF
    }
    cors_rule {
        allowed_origins = ["*"]
        allowed_methods = ["GET", "HEAD", "POST", "DELETE", "PUT"]
        allowed_headers = ["*"]
        expose_headers  = ["Date", "ETag"]}
        policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": ["s3:GetObject"],
            "Effect": "Allow",
            "Resource": "arn:aws:s3:::host-website-${var.env}/*",
            "Principal" : "*"
        }
    ]
}
EOF
}

resource "aws_cloudfront_distribution" "s3_distribution" {
  origin {
    domain_name = "host-website-${var.env}.s3.amazonaws.com"
    origin_id   = "host-website-${var.env}"
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"

  aliases = "${var.aliases}"

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "host-website-${var.env}"

    forwarded_values {
      query_string = false

      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  custom_error_response {
    error_code            = 400
    error_caching_min_ttl = 300
    response_code         = 200
    response_page_path    = "/404.html"
  }
  
  custom_error_response {
    error_code            = 403
    error_caching_min_ttl = 300
    response_code         = 200
    response_page_path    = "/404.html"
  }
  
  custom_error_response {
    error_code            = 404
    error_caching_min_ttl = 300
    response_code         = 200
    response_page_path    = "/404.html"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn  = "${var.acm_certificate_arn}"
    ssl_support_method = "sni-only"
  }
}