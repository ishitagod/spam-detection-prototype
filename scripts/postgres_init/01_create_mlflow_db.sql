-- POSTGRES_DB env var only creates feast_registry (the container's default
-- db) - this runs once on first container start to add the second database
-- MLflow's tracking store needs, in the same Postgres instance.
CREATE DATABASE mlflow;
