# Make sure to check bin/run_services.sh, which can be used here

# Do not forget to expose the right ports! (Check the PR_4.md)
FROM python:3.11-slim
WORKDIR /app

RUN pip install uv

#all the github code -> docker container
COPY . .

ENV PREFECT_API_URL="http://127.0.0.1:4201/api"

RUN uv sync

#need to give it this permission to execute
RUN chmod +x bin/run_services.sh

#for prefect and the fastapi
EXPOSE 4201
EXPOSE 8001

CMD ["./bin/run_services.sh"]
