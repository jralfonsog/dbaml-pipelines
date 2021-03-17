FROM databricksruntime/python:latest

RUN apt update \
  && apt install -y fuse \
  && apt clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV USER root

RUN mkdir /source

# Virtual environment

RUN virtualenv -p python3.7 --system-site-packages /databricks/python3
RUN /bin/bash -c "source /databricks/python3/bin/activate && pip install poetry==1.1.5"

COPY pyproject.toml poetry.lock ./

RUN /bin/bash -c """source /databricks/python3/bin/activate &&\
                    poetry install --no-dev --no-root"""

COPY source source/
RUN /bin/bash -c """source /databricks/python3/bin/activate &&\
                    poetry build && /databricks/python3/bin/pip install dist/*.whl"""