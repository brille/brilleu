FROM python:3-slim

LABEL "com.github.action.name"="Python3 pytest GitHub Action"
LABEL "com.github.actions.description"="Run pytest using python 3-slim image"
LABEL "com.github.actions.icon"="code"
LABEL "com.github.actions.color"="blue"

RUN pip install --upgrade pip pytest pytest-cov pytest-mock

WORKDIR /home
COPY entrypoint.sh /home
ENTRYPOINT ["sh","/home/entrypoint.sh"]
