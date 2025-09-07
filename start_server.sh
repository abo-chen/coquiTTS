#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 容器名称
CONTAINER_NAME="coqui-tts-gpu-service"
IMAGE_NAME="coqui-tts-gpu"

# 检查Docker是否在运行
check_docker() {
    if ! sudo docker info >/dev/null 2>&1; then
        echo -e "${RED}Docker is not running, please start Docker first${NC}"
        exit 1
    fi
}

# 停止并删除已存在的容器
stop_existing_container() {
    if sudo docker ps -a | grep -q "$CONTAINER_NAME"; then
        echo -e "${YELLOW}Stopping and removing existing container...${NC}"
        sudo docker stop $CONTAINER_NAME >/dev/null 2>&1
        sudo docker rm $CONTAINER_NAME >/dev/null 2>&1
    fi
}

# 构建镜像
build_image() {
    echo -e "${BLUE}Building Docker image...${NC}"
    # 检查是否存在 Dockerfile，如果没有则复制 Dockerfile.xtts
    if [ ! -f "Dockerfile" ] && [ -f "Dockerfile.xtts" ]; then
        cp Dockerfile.xtts Dockerfile
    fi
    
    sudo docker build -t $IMAGE_NAME . || {
        echo -e "${RED}Image build failed${NC}"
        exit 1
    }
    echo -e "${GREEN}Image build successful${NC}"
}

# 启动XTTS v2
start_xtts_v2() {
    echo -e "${BLUE}Starting XTTS v2 model...${NC}"
    stop_existing_container
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p 5002:5002 \
        -v $(realpath $(pwd)/../../tts-cache):/root/.local/share/tts \
        -v $(pwd)/tts-input:/root/tts-input \
        -v $(pwd)/tts-output:/root/tts-output \
        -v $(pwd)/TTS:/root/TTS \
        --gpus all \
        -e COQUI_TOS_AGREED=1 \
        --entrypoint python3 \
        ${IMAGE_NAME}:latest \
        TTS/server/server.py \
        --model_path /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 \
        --config_path /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json \
        --port 5002 --use_cuda true --show_details true
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}XTTS v2 started successfully${NC}"
        echo -e "${YELLOW}Waiting for service to fully start...${NC}"
        sleep 5
        echo -e "${GREEN}Service URL: http://localhost:5002${NC}"
    else
        echo -e "${RED}Start failed${NC}"
        exit 1
    fi
}

# 启动VCTK
start_vctk() {
    echo -e "${BLUE}Starting VCTK model...${NC}"
    stop_existing_container
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p 5002:5002 \
        -v $(realpath $(pwd)/../../tts-cache):/root/.local/share/tts \
        -v $(pwd)/tts-input:/root/tts-input \
        -v $(pwd)/tts-output:/root/tts-output \
        -v $(pwd)/TTS:/root/TTS \
        --gpus all \
        --entrypoint python3 \
        $IMAGE_NAME \
        TTS/server/server.py \
        --model_name tts_models/en/vctk/vits \
        --port 5002 --use_cuda=True --show_details=True
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}VCTK started successfully${NC}"
        echo -e "${GREEN}Service URL: http://localhost:5002${NC}"
    else
        echo -e "${RED}Start failed${NC}"
        exit 1
    fi
}

# 启动LJSpeech
start_ljspeech() {
    echo -e "${BLUE}Starting LJSpeech model...${NC}"
    stop_existing_container
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p 5002:5002 \
        -v $(realpath $(pwd)/../../tts-cache):/root/.local/share/tts \
        -v $(pwd)/tts-input:/root/tts-input \
        -v $(pwd)/tts-output:/root/tts-output \
        -v $(pwd)/TTS:/root/TTS \
        --gpus all \
        --entrypoint python3 \
        $IMAGE_NAME \
        TTS/server/server.py \
        --model_name tts_models/en/ljspeech/vits--neon \
        --port 5002 --use_cuda=True --show_details=True
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}LJSpeech started successfully${NC}"
        echo -e "${GREEN}Service URL: http://localhost:5002${NC}"
    else
        echo -e "${RED}Start failed${NC}"
        exit 1
    fi
}


# 启动Jenny
start_jenny() {
    echo -e "${BLUE}Starting Jenny model...${NC}"
    stop_existing_container
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p 5002:5002 \
        -v $(realpath $(pwd)/../../tts-cache):/root/.local/share/tts \
        -v $(pwd)/tts-input:/root/tts-input \
        -v $(pwd)/tts-output:/root/tts-output \
        -v $(pwd)/TTS:/root/TTS \
        --gpus all \
        --entrypoint python3 \
        $IMAGE_NAME \
        TTS/server/server.py \
        --model_name tts_models/en/jenny/jenny \
        --port 5002 --use_cuda=True --show_details=True
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Jenny started successfully${NC}"
        echo -e "${GREEN}Service URL: http://localhost:5002${NC}"
    else
        echo -e "${RED}Start failed${NC}"
        exit 1
    fi
}

# 下载XTTS v2模型
download_xtts_v2() {
    echo -e "${BLUE}Downloading XTTS v2 model...${NC}"
    echo -e "${YELLOW}This may take several minutes, please wait...${NC}"
    
    sudo docker run -it --rm \
        -v $(realpath $(pwd)/../../tts-cache):/root/.local/share/tts \
        -e COQUI_TOS_AGREED=1 \
        $IMAGE_NAME:latest \
        --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
        --text "hello world" \
        --speaker_id "Claribel Dervla" \
        --language_idx "en"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}XTTS v2 model downloaded successfully${NC}"
        echo -e "${YELLOW}Press any key to return to menu...${NC}"
        read -n 1
    else
        echo -e "${RED}Model download failed${NC}"
        echo -e "${YELLOW}Press any key to return to menu...${NC}"
        read -n 1
    fi
}

# 使用docker-compose启动
start_with_compose() {
    echo -e "${BLUE}Starting with docker-compose...${NC}"
    
    # 确保tts-input和tts-output目录存在
    mkdir -p tts-input tts-output
    
    # 停止已存在的容器
    sudo docker-compose down 2>/dev/null
    
    # 构建并启动
    sudo docker-compose up -d --build
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Service started successfully${NC}"
        echo -e "${YELLOW}Waiting for service to fully start...${NC}"
        sleep 5
        echo -e "${GREEN}Service URL: http://localhost:5002${NC}"
        echo -e "${YELLOW}View logs: sudo docker-compose logs -f${NC}"
    else
        echo -e "${RED}Start failed${NC}"
        exit 1
    fi
}

# 显示容器状态
show_status() {
    echo -e "${BLUE}Container Status:${NC}"
    sudo docker ps -a | grep -E "CONTAINER|$CONTAINER_NAME" --color=never
    echo ""
    if sudo docker ps | grep -q "$CONTAINER_NAME"; then
        echo -e "${GREEN}Service running: http://localhost:5002${NC}"
        echo -e "${YELLOW}View logs: sudo docker logs -f $CONTAINER_NAME${NC}"
    else
        echo -e "${RED}Service not running${NC}"
    fi
}

# 查看实时日志
show_logs() {
    if sudo docker ps | grep -q "$CONTAINER_NAME"; then
        echo -e "${BLUE}Showing live logs (Press Ctrl+C to exit)...${NC}"
        echo -e "${YELLOW}========================================${NC}"
        sudo docker logs -f $CONTAINER_NAME
    else
        echo -e "${RED}Container is not running${NC}"
        echo -e "${YELLOW}Press any key to return to menu...${NC}"
        read -n 1
    fi
}

# 主菜单
show_menu() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}    Coqui TTS Docker Manager${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "  === TTS Models ==="
    echo "  1) Start XTTS v2 (Multilingual, Recommended)"
    echo "  2) Start VCTK (English, Multi-speaker)"
    echo "  3) Start LJSpeech (English, Single speaker)"
    echo "  a) Start Jenny (English, Single speaker)"
    echo ""
    echo "  === Management ==="
    echo "  4) Download XTTS v2 Model"
    echo "  5) Start with docker-compose"
    echo "  6) Build Docker Image"
    echo "  7) Check Container Status"
    echo "  8) View Live Logs (Ctrl+C to exit)"
    echo "  9) Stop Service"
    echo "  0) Exit"
    echo ""
    echo -e "${YELLOW}========================================${NC}"
}

# 检查必要条件
check_docker

# 确保必要目录存在
mkdir -p tts-input tts-output

# 主循环
while true; do
    show_menu
    read -p "Please select an option [0-9,a]: " choice
    
    case $choice in
        1)
            if [ ! "$(sudo docker images -q $IMAGE_NAME 2>/dev/null)" ]; then
                echo -e "${YELLOW}Image not found, building first...${NC}"
                build_image
            fi
            start_xtts_v2
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        2)
            if [ ! "$(sudo docker images -q $IMAGE_NAME 2>/dev/null)" ]; then
                echo -e "${YELLOW}Image not found, building first...${NC}"
                build_image
            fi
            start_vctk
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        3)
            if [ ! "$(sudo docker images -q $IMAGE_NAME 2>/dev/null)" ]; then
                echo -e "${YELLOW}Image not found, building first...${NC}"
                build_image
            fi
            start_ljspeech
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        4)
            download_xtts_v2
            ;;
        5)
            start_with_compose
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        6)
            build_image
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        7)
            show_status
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        8)
            show_logs
            ;;
        9)
            stop_existing_container
            echo -e "${GREEN}Service stopped${NC}"
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        a|A)
            if [ ! "$(sudo docker images -q $IMAGE_NAME 2>/dev/null)" ]; then
                echo -e "${YELLOW}Image not found, building first...${NC}"
                build_image
            fi
            start_jenny
            echo -e "${YELLOW}Press any key to return to menu...${NC}"
            read -n 1
            ;;
        0)
            echo -e "${GREEN}Exiting program${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid selection, please try again${NC}"
            sleep 2
            ;;
    esac
done