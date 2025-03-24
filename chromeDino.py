import numpy as np
from PIL import Image

import cv2
import io
import time
import random
import pickle
import os
from io import BytesIO
import base64
import json
import pandas as pd

from collections import deque
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys

from webdriver_manager.chrome import ChromeDriverManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# GAME_URL = "chrome://dino"

# 초기 환경 설정
GAME_URL = "https://fivesjs.skipser.com/trex-game1/"
CHROME_DRIVER_PATH = ChromeDriverManager().install()

DATA_DIR = "./data"
MODEL_DIR = "./model"
SAVE_INTERVAL = 1000

device = torch.device("cuda:0")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS_FILE = os.path.join(DATA_DIR, "params.pkl")

INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
GET_BASE64_SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"


def save_params(params):
    with open(PARAMS_FILE, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

def load_params():
    if os.path.isfile(PARAMS_FILE):
        with open(PARAMS_FILE, 'rb') as f:
            print("파라미터 파일 존재!")
            params = pickle.load(f)
            return params
    print("초기 파라미터 사용!")
    return {
        "D": deque(maxlen=50000),
        "time": 0,
        "epsilon": 0.01
    }

def load_model():
    model = DinoNet()  # DinoNet 모델을 먼저 초기화
    model_path = './latest.pth'

    if os.path.isfile(model_path):
        try:
            print("모델 불러오기 성공!")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # CPU에서도 로드 가능
            model.eval()  # 평가 모드로 전환
        except Exception as e:
            print(f"모델 불러오기 실패: {e}")
            print("새로운 모델로 시작합니다.")
    else:
        print("저장된 모델이 없습니다. 새로운 모델로 시작합니다.")

    return model

#이미지를 캡쳐
def grab_screen(driver):
    image_b64 = driver.execute_script(GET_BASE64_SCRIPT)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    return process_img(screen)

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:300, :500]
    image = cv2.resize(image, (80, 80))
    return image

def show_img(graphs=False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, imS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# 현재 게임의 상태를 알기 위한 부분
class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        service = Service(CHROME_DRIVER_PATH)
        self._driver = webdriver.Chrome(service=service, options=chrome_options)
        self._driver.set_window_position(x=300, y=300)
        self._driver.set_window_size(900, 600)
        
        try : 
            self._driver.get(GAME_URL)
        except:
            pass
        
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(INIT_SCRIPT)
    
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    
    def press_up(self):
        self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_UP)
    
    def press_down(self):
        self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_DOWN)
    
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        return int(''.join(score_array))
    
    def pause(self):
        self._driver.execute_script("return Runner.instance_.stop()")
    
    def resume(self):
        self._driver.execute_script("return Runner.instance_.play()")
    
    def end(self):
        self._driver.close()

# 게임을 플레이하는 agent
class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump()
    
    def is_running(self):
        return self._game.get_playing()
    
    def is_crashed(self):
        return self._game.get_crashed()
    
    def jump(self):
        self._game.press_up()
    
    def duck(self):
        self._game.press_down()

# state: 현재 state와 action에 따른 reward 설정
class GameState:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__()
    
    def get_state(self, actions):
        score = self._game.get_score()  # 기본적으로 아무것도 안할 때
        reward = 0.1
        is_over = False
        
        if actions[1] == 1: # action == jump
            self._agent.jump()
            reward = -0.0005

        image = grab_screen(self._game._driver) # 현재 화면 출력
        self._display.send(image)
        
        if self._agent.is_crashed():    # game over(terminal)
            self._game.restart()
            reward = -100
            is_over = True
        
        return image, reward, is_over
    
ACTIONS = 2
GAMMA = 0.99
OBSERVATION = 1000  
EXPLORE = 500000
FINAL_EPSILON = 0.00001
INITIAL_EPSILON = 0.1  
REPLAY_MEMORY = 100000
BATCH_SIZE = 32  
LEARNING_RATE = 1e-5
IMG_CHANNELS = 4

# CNN: 캡쳐된 화면을 통해 [가만히 있기, 점프]를 출력한다
class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
    
    # forward propagation
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   #(batch, height, width, channels) -> (batch, channels, height, width)
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.max_pool2d(self.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# 학습
def train_network(model, game_state, observe=False, params=None):
    # 파라미터 로드
    D = params["D"]     # experience를 저장하는 queue
    t = params["time"]  # time step
    epsilon = params["epsilon"] # exploration / exploitation

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    # CNN 신경망 학습 진행
    loss_fn = nn.MSELoss()

    do_nothing = np.zeros(2)
    do_nothing[0] = 1   # [1,0], 아무것도 안함
    
    # 아무것도 안하는 행동을 취한 후 state를 가져옴
    x_t, r_0, terminal = game_state.get_state(do_nothing)   # x_t: 현재 게임의 이미지, r_0: 현재 게임의 보상
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 이전 몇개 프레임과 결합

    OBSERVE = 999999999 if observe else 100

    while True:
        loss_sum = 0        # loss
        a_t = np.zeros([2]) # action    [1, 0] = 아무것도 안하기, [0, 1] = 점프

        # e-greedy
        if random.random() / 1000 <= epsilon:   # epsilon의 확률로 랜덤한 행동을 시행
            action_index = random.randrange(2)  # 1로 활성화할 위치 랜덤으로 정함
            a_t[action_index] = 1
            print("++Explore!++", end='')
        else:
            q = model(torch.tensor(s_t).float())    # 현재 상태에서 q값(action-value function)을 예측
            _, action_index = torch.max(q, 1)       # q값이 가장 큰 값을 선택
            action_index = action_index.item()
            a_t[action_index] = 1
            print("__Exploit!__", end='')

        # decaying epsilon algorithm
        if epsilon > FINAL_EPSILON and t > OBSERVE:     # OBSERVE: 
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        # 현재 선택된 행동 a_t를 게임에 적용 후 이후의 상태 확인
        x_t1, r_t, terminal = game_state.get_state(a_t)         # x_t1: action 이후 현재 게임의 이미지, r_t: 행동을 통해 얻은 보상
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)        # 이전 몇개 프레임과 결합

        # experience를 저장하는 queue
        if len(D) > 50000:
            D.pop()
        D.append((s_t, action_index, r_t, s_t1, terminal))

        if t > OBSERVE:
            minibatch = random.sample(D, 16)    # experience D에서 16개 샘플링
            inputs = np.zeros((16, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((16, 2))

            for i in range(16):
                state_t, action_t, reward_t, state_t1, terminal = minibatch[i]  # 한 샘플에서 s_t, a_t, r_t, s_t1, 종료 여부를 가져옴
                inputs[i:i + 1] = state_t                                       # 전체 minibatch의 상태 데이터를 담고 있음
                target = model(torch.tensor(state_t).float()).detach().numpy()[0]   # s_t에서 신경망(CNN)을 이용해 q값을 예측(terminal인 경우 사용)
                Q_sa = model(torch.tensor(state_t1).float()).detach().numpy()[0]    # s_t1에서 신경망(CNN)을 이용해 q값을 예측

                # Bellman Eq.
                if terminal:    # terminal인 경우
                    target[action_t] = reward_t
                else:           # 아닌 경우 q값 사용해서 업데이트
                    target[action_t] = reward_t + GAMMA * np.max(Q_sa)  # reward + GAMMA(Q(S, A)), discount factor = 0.99, 미래의 보상을 가치있게 판단

                targets[i] = target

            outputs = model(torch.tensor(inputs).float())       # 모델(DinoNet, CNN)이 예측한 현재 상태에서 선택한 행동의 q값
            loss = loss_fn(outputs, torch.tensor(targets).float())  # Bellman Eq.에 의해 계산된 목표 q값

            optimizer.zero_grad() # gradient 초기화(pytorch에서는 기울기가 누적, 계산된 기울기가 다음 연산에 영향을 미칠 수 있으므로 초기화)
            loss.backward()         # backpropagation을 수행, gradient 계산
            optimizer.step()        # 기울기 기반으로 가중치 업데이트

            loss_sum += loss.item()

        # terminal인 경우 s_t로 업데이트 아니라면 다음 state s_t1
        s_t = s_t1 if not terminal else s_t
        t += 1

        # 모델 저장 코드
        if t % SAVE_INTERVAL == 0:
            game_state._game.pause()
            print(os.path.join(MODEL_DIR, f"episode_{t}.pth"))
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"episode_{t}.pth"))
            torch.save(model.state_dict(), "./latest.pth")
            save_params({"D": D, "time": t, "epsilon": epsilon})
            game_state._game.resume()

        print(f'timestep: {t}, epsilon: {epsilon}, action: {action_index}, reward: {r_t}, loss: {round(loss_sum, 3)}')



def play_game(observe=False):
    game = Game()
    agent = DinoAgent(game)
    game_state = GameState(agent, game)
    

    try:
        model = load_model()
        params = load_params()
        print("이전 모델 불러오기 완료!")
        train_network(model, game_state, observe, params)
    except StopIteration:
        game.end()

        
        
play_game(observe=False)