# Step 1: Build the React application
FROM node:20 AS build

# ARG VITE_BACKEND_API_URL="http://172.18.0.2:4000"
ARG VITE_BACKEND_API_URL="http://117.50.174.65/kgbuilderapi"
ARG VITE_REACT_APP_SOURCES=""
ARG VITE_GOOGLE_CLIENT_ID=""
ARG VITE_BLOOM_URL="https://workspace-preview.neo4j.io/workspace/explore?connectURL={CONNECT_URL}&search=Show+me+a+graph&featureGenAISuggestions=true&featureGenAISuggestionsInternal=true"
ARG VITE_TIME_PER_PAGE=50
ARG VITE_LARGE_FILE_SIZE=5242880
ARG VITE_CHUNK_SIZE=5242880
ARG VITE_CHAT_MODES=""
ARG VITE_ENV="DEV"
ARG VITE_BATCH_SIZE=2
ARG VITE_LLM_MODELS="openai_gpt_4o,openai_gpt_4o_mini,baidu_ernie_4_0,xunfei_spark_4_0,tengxun_hunyuan_pro,ali_tongyi_qwen_max,ollama_qwen_2_5,vllm_minicpm_3,ollama_deepseek_r_1,siliconflow_deepseek_r_1,siliconflow_deepseek_v_3"
ARG VITE_LLM_MODELS_PROD="openai_gpt_4o,openai_gpt_4o_mini,baidu_ernie_4_0,xunfei_spark_4_0,tengxun_hunyuan_pro,ali_tongyi_qwen_max,ollama_qwen_2_5,vllm_minicpm_3,ollama_deepseek_r_1,siliconflow_deepseek_r_1,siliconflow_deepseek_v_3"

RUN apt-get update && apt-get install -y \
    curl \
    vim \
    net-tools \
    iputils-ping \
    inetutils-telnet \
    apt-utils \
    && apt-get clean

WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install
COPY . ./
RUN VITE_BACKEND_API_URL=$VITE_BACKEND_API_URL \
    VITE_REACT_APP_SOURCES=$VITE_REACT_APP_SOURCES \
    VITE_GOOGLE_CLIENT_ID=$VITE_GOOGLE_CLIENT_ID \
    VITE_BLOOM_URL=$VITE_BLOOM_URL \
    VITE_CHUNK_SIZE=$VITE_CHUNK_SIZE \
    VITE_TIME_PER_PAGE=$VITE_TIME_PER_PAGE \
    VITE_ENV=$VITE_ENV \
    VITE_LARGE_FILE_SIZE=${VITE_LARGE_FILE_SIZE} \
    VITE_CHAT_MODES=$VITE_CHAT_MODES \
    VITE_BATCH_SIZE=$VITE_BATCH_SIZE \
    VITE_LLM_MODELS=$VITE_LLM_MODELS \
    VITE_LLM_MODELS_PROD=$VITE_LLM_MODELS_PROD \
    yarn run build

# Step 2: Serve the application using Nginx
FROM nginx:alpine
ARG DEPLOYMENT_ENV="local"
ENV DEPLOYMENT_ENV=$DEPLOYMENT_ENV
COPY --from=build /app/dist /usr/share/nginx/html
COPY /nginx/nginx.${DEPLOYMENT_ENV}.conf /etc/nginx/templates/nginx.conf.template

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
