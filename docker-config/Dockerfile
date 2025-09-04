# Use the official PostgreSQL image as base
FROM postgres:15

# Set environment variables for the default DB, user, and password
ENV POSTGRES_DB=mydatabase
ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword

# Copy the initial SQL script into Docker's init directory
# PostgreSQL image automatically runs scripts in /docker-entrypoint-initdb.d
COPY init.sql /docker-entrypoint-initdb.d/
