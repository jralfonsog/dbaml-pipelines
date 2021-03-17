include .env
export

.PHONY: help
.PHONY: test

help: ## Shows this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build-image: ## Builds image with source code
	docker build . --tag ${CONTAINER_REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_VERSION}

test-image: ## Test image locally from Terminal
	docker run -it ${CONTAINER_REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_VERSION} /bin/bash

push-image: ## Push image to ACR
	az acr login --name ${CONTAINER_REGISTRY_NAME} \
	&& docker push ${CONTAINER_REGISTRY_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_VERSION}