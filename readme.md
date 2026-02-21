To train locally, set env variables


export AWS_ACCESS_KEY_ID=XXX
export AWS_SECRET_ACCESS_KEY=XXX
export AWS_DEFAULT_REGION=ap-south-1



# ETL -> When new data arrives
# Feast Apply -> When feature changes
# Feast Materialize -> After new data
# Drift -> Daily scheduled jobs
# DVC Push -> After successful pipeline

To start FastAPI service
uvicorn app:app --reload

