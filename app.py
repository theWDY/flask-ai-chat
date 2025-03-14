import time  # 导入时间模块，用于时间处理
from flask import Flask, request, jsonify, render_template, Response, session
import requests  # 导入requests库，用于发送HTTP请求
from openai import OpenAI  # 导入OpenAI客户端，用于与Moonshot API进行交互
import os  # 导入操作系统模块，用于访问环境变量等
import json  # 导入JSON模块，用于处理JSON数据
from typing import *  # 导入所有类型，用于类型注解
from pathlib import Path  # 导入Path类，用于文件路径操作
import concurrent.futures  # 导入并发模块，用于执行并发任务,需要好好学一下
from typing import List, Dict  # 导入List和Dict类型，用于类型注解
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from functools import wraps
import re  # 添加正则表达式支持
import smtplib  # 添加邮件发送支持
from email.mime.text import MIMEText  # 添加邮件内容支持
from email.header import Header  # 添加邮件头部支持
import random  # 添加随机数支持
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

app = Flask(__name__)  # 创建Flask应用实例
app.secret_key = os.urandom(24)  # 设置session密钥

# 邮件配置
SMTP_SERVER = "smtp.qq.com"  # QQ邮箱SMTP服务器
SMTP_PORT = 465  # 修改为SSL端口
SMTP_USER = os.environ.get("SMTP_USER", "3295123703@qq.com")  # 发件人邮箱
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "你的QQ邮箱授权码")  # 发件人邮箱授权码

# 检查邮箱配置
if not SMTP_USER or not SMTP_PASSWORD:
    print("警告: 邮箱配置未设置！请设置 SMTP_USER 和 SMTP_PASSWORD 环境变量")
    print("SMTP_USER 应该是您的QQ邮箱地址")
    print("SMTP_PASSWORD 应该是您的QQ邮箱SMTP授权码")

# 验证码有效期（秒）
VERIFICATION_CODE_EXPIRY = 300  # 5分钟

# 存储验证码的字典 {email: (code, timestamp)}
verification_codes = {}

def send_verification_email(to_email: str, code: str) -> bool:
    """
    发送验证码邮件，
    :param to_email: 收件人邮箱
    :param code: 验证码
    :return: 是否发送成功
    """
    try:
        msg = MIMEText(f'您的验证码是：{code}，有效期5分钟。请勿将验证码泄露给他人。', 'plain', 'utf-8')
        msg['From'] = f'{SMTP_USER}'  # 直接使用邮箱地址
        msg['To'] = to_email
        msg['Subject'] = '验证码 - AI助手'

        # 使用SSL连接
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"发送邮件失败: {str(e)}")
        return False

def generate_verification_code() -> str:
    """
    生成6位数字验证码
    """
    return ''.join(random.choices('0123456789', k=6))

def is_valid_email(email: str) -> bool:
    """
    验证邮箱格式是否正确
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

@app.route('/send_verification_code', methods=['POST'])
def send_verification_code():
    """
    发送验证码接口
    """
    try:
        # 检查邮箱配置
        if not SMTP_USER or not SMTP_PASSWORD:
            return jsonify({
                'message': '邮箱服务未配置，请联系管理员设置SMTP_USER和SMTP_PASSWORD环境变量'
            }), 500

        data = request.get_json()
        email = data.get('email')

        if not email:
            return jsonify({'message': '邮箱不能为空'}), 400

        if not is_valid_email(email):
            return jsonify({'message': '邮箱格式不正确'}), 400

        # 检查邮箱是否已被注册
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        c.execute('SELECT email FROM users WHERE email = ?', (email,))
        if c.fetchone():
            conn.close()
            return jsonify({'message': '该邮箱已被注册'}), 400
        conn.close()

        # 生成验证码
        code = generate_verification_code()
        
        # 发送验证码邮件
        if send_verification_email(email, code):
            # 存储验证码和时间戳
            verification_codes[email] = (code, time.time())
            return jsonify({'message': '验证码已发送'}), 200
        else:
            return jsonify({'message': '验证码发送失败，请确认邮箱地址是否正确'}), 500

    except Exception as e:
        print(f"发送验证码失败: {str(e)}")
        return jsonify({'message': '发送验证码失败，请稍后重试'}), 500

kimi_client = OpenAI(  # 创建一个OpenAI客户端实例
    api_key=os.environ["MOONSHOT_DEMO_API_KEY"],  # 从环境变量获取API密钥
    base_url="https://api.moonshot.cn/v1",  # 设置API的基础URL
)

deepseek_client = OpenAI(  # 创建一个OpenAI客户端实例
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 从环境变量获取API密钥
    base_url="https://api.deepseek.com/v1",  # 设置API的基础URL
)

client = "kimi_client"

MODEL_TO_CLIENT = {
    # Kimi模型
    "moonshot-v1-8k": kimi_client,
    "moonshot-v1-32k": kimi_client,
    "moonshot-v1-128k": kimi_client,
    # DeepSeek模型
    "deepseek-chat": deepseek_client,
    "deepseek-reasoner": deepseek_client,
}

start_message="你是由WDY通过集成api所提供的人工智能助手，你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。"

history_index = 0  # 初始化对话历史索引，默认为0

history = []  # 用来存储对话历史，初始为空列表

# 用于控制是否中止请求的标志
is_stopped = False  # 初始化中止标志为False

filespathes = []  # 存储所有上传文件的绝对路径，初始为空列表

length = 0  # 当前对话已上传的文件数量

add_files = 0  # 本次需要新上传的文件数量，初始为0

# 数据库初始化
def init_db():
    conn = None
    try:
        conn = sqlite3.connect('users.db', timeout=20)  # 添加超时设置
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                avatar_path TEXT
            )
        ''')
        conn.commit()
    except Exception as e:
        print(f"数据库初始化错误: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# 在应用启动时初始化数据库
with app.app_context():
    init_db()

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'message': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

# 密码强度验证函数
def validate_password_strength(password):
    """
    验证密码强度
    返回 (bool, str) 元组，表示是否通过验证和错误信息
    """
    if len(password) < 8:
        return False, "密码长度必须至少为8个字符"
    if not re.search(r"[A-Z]", password):
        return False, "密码必须包含至少一个大写字母"
    if not re.search(r"[a-z]", password):
        return False, "密码必须包含至少一个小写字母"
    if not re.search(r"\d", password):
        return False, "密码必须包含至少一个数字"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "密码必须包含至少一个特殊字符"
    return True, ""

# 注册路由
@app.route('/register', methods=['POST'])
def register():
    conn = None
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        verification_code = data.get('verification_code')

        # 检查所有必填字段
        if not all([username, email, password, verification_code]):
            print(f"\n注册失败 - 原因：字段不完整")
            print(f"用户名: {username or '空'}")
            print(f"邮箱: {email or '空'}")
            print(f"密码: {'已填写' if password else '空'}")
            print(f"验证码: {verification_code or '空'}\n")
            return jsonify({'message': '所有字段都是必填的'}), 400

        # 验证邮箱格式
        if not is_valid_email(email):
            print(f"\n注册失败 - 原因：邮箱格式不正确")
            print(f"邮箱: {email}\n")
            return jsonify({'message': '邮箱格式不正确'}), 400

        # 验证验证码
        if email not in verification_codes:
            print(f"\n注册失败 - 原因：未发送验证码")
            print(f"邮箱: {email}\n")
            return jsonify({'message': '请先获取验证码'}), 400

        stored_code, timestamp = verification_codes[email]
        if time.time() - timestamp > VERIFICATION_CODE_EXPIRY:
            # 验证码过期，删除记录
            del verification_codes[email]
            print(f"\n注册失败 - 原因：验证码已过期")
            print(f"邮箱: {email}\n")
            return jsonify({'message': '验证码已过期，请重新获取'}), 400

        if verification_code != stored_code:
            print(f"\n注册失败 - 原因：验证码错误")
            print(f"邮箱: {email}\n")
            return jsonify({'message': '验证码错误'}), 400

        # 验证密码强度
        is_valid, error_message = validate_password_strength(password)
        if not is_valid:
            print(f"\n注册失败 - 原因：密码强度不足")
            print(f"错误信息: {error_message}\n")
            return jsonify({'message': error_message}), 400

        # 密码加密
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        
        # 先检查用户名是否存在
        c.execute('SELECT username FROM users WHERE username = ?', (username,))
        if c.fetchone():
            print(f"\n注册失败 - 原因：用户名已存在")
            print(f"用户名: {username}\n")
            return jsonify({'message': '用户名已存在'}), 400
            
        # 检查邮箱是否存在
        c.execute('SELECT email FROM users WHERE email = ?', (email,))
        if c.fetchone():
            print(f"\n注册失败 - 原因：邮箱已存在")
            print(f"邮箱: {email}\n")
            return jsonify({'message': '邮箱已存在'}), 400

        # 插入新用户
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                 (username, email, hashed_password))
        conn.commit()

        # 注册成功后删除验证码记录
        del verification_codes[email]

        print(f"\n注册成功！")
        print(f"用户名: {username}")
        print(f"邮箱: {email}")
        print(f"密码: {password}\n")
        return jsonify({'message': '注册成功'}), 201

    except sqlite3.IntegrityError as e:
        print(f"\n注册失败 - 原因：数据完整性错误 {str(e)}")
        print(f"用户名: {username}")
        print(f"邮箱: {email}\n")
        if conn:
            conn.rollback()
        return jsonify({'message': '用户名或邮箱已存在'}), 400

    except Exception as e:
        print(f"\n注册失败 - 原因：数据库错误 {str(e)}")
        print(f"用户名: {username}")
        print(f"邮箱: {email}\n")
        if conn:
            conn.rollback()
        return jsonify({'message': '注册失败，请稍后重试'}), 500

    finally:
        if conn:
            conn.close()

# 登录路由
@app.route('/login', methods=['POST'])
def login():
    conn = None
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        # 检查必填字段
        if not all([username, password]):
            print(f"\n登录失败 - 原因：字段不完整")
            print(f"用户名: {username or '空'}")
            print(f"密码: {'已填写' if password else '空'}\n")
            return jsonify({'message': '用户名和密码都是必填的'}), 400

        conn = sqlite3.connect('users.db', timeout=20)  # 添加超时设置
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            print(f"\n登录成功！")
            print(f"用户名: {user[1]}")
            print(f"邮箱: {user[2]}")
            print(f"密码: {password}\n")
            return jsonify({
                'message': '登录成功',
                'username': user[1]
            }), 200
        else:
            if not user:
                print(f"\n登录失败 - 原因：用户名不存在")
                print(f"用户名: {username}\n")
            else:
                print(f"\n登录失败 - 原因：密码错误")
                print(f"用户名: {username}")
                print(f"邮箱: {user[2]}\n")
            return jsonify({'message': '用户名或密码错误'}), 401

    except Exception as e:
        print(f"\n登录失败 - 原因：数据库错误 {str(e)}")
        print(f"用户名: {username}\n")
        return jsonify({'message': '登录失败，请稍后重试'}), 500

    finally:
        if conn:
            conn.close()

# 退出登录路由
@app.route('/logout', methods=['POST'])
def logout():
    try:
        # 在清除session前获取用户信息
        username = session.get('username')
        
        # 获取用户的完整信息
        if username:
            conn = sqlite3.connect('users.db', timeout=20)
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            conn.close()
            
            if user:
                print(f"\n退出登录！")
                print(f"用户名: {user[1]}")
                print(f"邮箱: {user[2]}\n")
        
        # 清除session
        session.clear()
        return jsonify({'message': '退出成功'}), 200
    except Exception as e:
        print(f"\n退出登录失败 - 原因：{str(e)}")
        return jsonify({'message': '退出失败，请稍后重试'}), 500

# 获取当前用户信息路由
@app.route('/user/current', methods=['GET'])
@login_required
def get_current_user():
    return jsonify({
        'username': session.get('username'),
        'user_id': session.get('user_id')
    })

def handle_interruption(original_index):
    """处理中断和对话索引变化的情况"""
    global history, history_index, is_stopped
    if is_stopped or original_index != history_index:
        if is_stopped:
            print(f"handle_interruption: 返回True，因为is_stopped为True")
            return True
        else:
            cleanup_incomplete_message(original_index)
            print(f"handle_interruption: 返回None，因为索引变化")
            return None
    
    print(f"handle_interruption: 返回None，正常执行")
    return None


def cleanup_incomplete_message(index):
    """清理未完成的用户消息"""
    global history
    if len(history[index]) > 0 and history[index][-1].get("role") == "user":
        history[index].pop()
    print(f"cleanup_incomplete_message: 执行完毕，处理索引 {index}")


#需要检查判断条件对deepseek是否有效
def handle_api_error(e, original_index):
    """处理API调用时的常见错误"""
    error_msg = str(e).lower()
    # 确保索引有效
    if original_index >= len(history):
        original_index = len(history) - 1 if len(history) > 0 else 0

    if "rate_limit" in error_msg:
        print(f"handle_api_error: 返回频率限制错误")
        return "请求过于频繁，请稍后重试"
    elif any(phrase in error_msg for phrase in ["exceeded model token limit", "context window", "context_length"]):
        print(f"handle_api_error: 返回上下文长度错误")
        return "当前对话上下文过长，请选择有更长上下文的模型或开启新对话"
    else:
        print(f"handle_api_error: 抛出未知错误 {str(e)}")
        raise e


def append_message(message, original_index):
    """将消息添加到历史记录中"""
    global history, history_index, client
    # 确保索引有效
    if original_index >= len(history):
        # 如果索引无效，创建新的对话历史
        while len(history) <= original_index:
            history.append([])
            if len(history[-1]) == 0:
                # 为新对话添加系统提示信息
                history[-1].append({
                    "role": "system",
                    "content": start_message
                })
        # 更新history_index为新对话的索引
        history_index = original_index
    
    # 添加消息到指定的对话历史中
    history[original_index].append(message)
    print(f"append_message: 执行完毕，添加消息到索引 {original_index}，消息类型 {message.get('role')}")
    print(f"append_message: 当前history_index为 {history_index}")


def chatbot(query, model, search_toggle, message_index=None):
    global filespathes, history_index, history, length, add_files, is_stopped, client

    original_index = history_index

    client = MODEL_TO_CLIENT.get(model, kimi_client)

    try:
        # 如果是编辑历史消息，先删除该消息之后的所有对话记录，但保留系统消息
        if message_index is not None:
            # 保存系统消息
            system_messages = [msg for msg in history[original_index] if msg["role"] == "system"]
            # 保存到message_index的用户和助手消息
            other_messages = [msg for msg in history[original_index][:message_index + 1] if msg["role"] != "system"]
            # 重建历史记录
            history[original_index] = system_messages + other_messages
            # 添加新的用户消息
            append_message({"role": "user", "content": query}, original_index)
        
        if add_files > 0:
            for index in range(add_files):
                interruption = handle_interruption(original_index)
                if interruption:
                    print(f"chatbot: 文件处理中断")
                    return
                try:
                    file_object = client.files.create(file=Path(filespathes[original_index][length + index]),
                                                      purpose="file-extract")
                    file_content = client.files.content(file_id=file_object.id).text
                    append_message({"role": "system", "content": file_content}, original_index)
                except Exception as e:
                    error_msg = handle_api_error(e, original_index)
                    yield error_msg
                    print(f"chatbot: 文件处理错误")
                    return
            add_files = 0

        interruption = handle_interruption(original_index)
        if interruption:
            print(f"chatbot: 对话中断")
            return

        if is_stopped:
            print(f"chatbot: 对话停止")
            return

        if search_toggle == 'no':
            try:
                # 如果是deepseek-reasoner模型，过滤掉中断消息
                messages_to_send = history[original_index]
                if model == "deepseek-reasoner":
                    # 只过滤掉"回答已被中断"的消息
                    filtered_messages = [msg for msg in history[original_index] if not (
                        msg.get("role") == "assistant" and 
                        msg.get("content") == "回答已被中断"
                    )]
                    messages_to_send = filtered_messages

                stream = client.chat.completions.create(
                    model=f"{model}",
                    messages=messages_to_send,
                    temperature=0.3,
                    stream=True,
                )
                
                collectedMessage = ""
                for chunk in stream:
                    if is_stopped:
                        if collectedMessage:
                            append_message({"role": "assistant", "content": collectedMessage}, original_index)
                            append_message({"role": "assistant", "content": "回答已被中断"}, original_index)
                        print(f"chatbot: 流式回复中断")
                        return
                    
                    # 处理推理模型的特殊输出
                    if model == "deepseek-reasoner":
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            collectedMessage += chunk.choices[0].delta.reasoning_content
                            yield chunk.choices[0].delta.reasoning_content
                        elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            collectedMessage += chunk.choices[0].delta.content
                            yield chunk.choices[0].delta.content
                    # 处理普通模型的输出
                    elif chunk.choices[0].delta.content is not None:
                        collectedMessage += chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                    
                # 在流式响应结束后添加完整消息
                if collectedMessage and not is_stopped:
                    append_message({"role": "assistant", "content": collectedMessage}, original_index)
                    
            except Exception as e:
                error_msg = handle_api_error(e, original_index)
                yield error_msg
                print(f"chatbot: API调用错误")
                return
        else:
            try:
                collectedMessage = ""
                for chunk in linked_chatbot(model):
                    if chunk:
                        collectedMessage += chunk
                        yield chunk
                    if is_stopped and collectedMessage:
                        append_message({"role": "assistant", "content": collectedMessage}, original_index)
                        append_message({"role": "assistant", "content": "回答已被中断"}, original_index)
                        print(f"chatbot: 联网回复中断")
                        return
            except Exception as e:
                error_msg = handle_api_error(e, original_index)
                yield error_msg
                print(f"chatbot: 联网调用错误")
                return

    except Exception as e:
        error_msg = str(e)
        yield error_msg
        print(f"chatbot: 未知错误 {str(e)}")
        return

    print(f"chatbot: 执行完毕")


def search_impl(arguments: Dict[str, Any]) -> Any:  # 定义 search 工具的具体实现，参数为一个字典类型的 arguments
    """
    在使用 Moonshot AI 提供的 search 工具的场合，只需要原封不动返回 arguments 即可，
    不需要额外的处理逻辑。

    但如果你想使用其他模型，并保留联网搜索的功能，那你只需要修改这里的实现（例如调用搜索
    和获取网页内容等），函数签名不变，依然是 work 的。

    这最大程度保证了兼容性，允许你在不同的模型间切换，并且不需要对代码有破坏性修改。
    """
    return arguments  # 直接返回 arguments，表示不进行任何额外处理


#deepseek模型无法使用搜索功能
def pre_link(model):
    global history, history_index

    # 获取当前对话历史
    current_history = history[history_index]
    
    # 分离系统消息和其他消息
    system_messages = [msg for msg in current_history if msg["role"] == "system"]
    user_messages = [msg for msg in current_history if msg["role"] == "user"]
    assistant_messages = [msg for msg in current_history if msg["role"] == "assistant"]
    tool_messages = [msg for msg in current_history if msg["role"] == "tool"]


    
    # 重新组织消息顺序
    messages = []
    messages.extend(system_messages)  # 系统消息放在最前面
    
    # 添加最后一轮对话相关的所有消息
    if user_messages:
        last_user_msg = user_messages[-1]
        messages.append(last_user_msg)
        
        # 找到这个用户消息之后的所有tool_calls和tool响应
        user_msg_index = current_history.index(last_user_msg)
        for msg in current_history[user_msg_index:]:
            if msg.get("tool_calls") or msg["role"] == "tool":
                messages.append(msg)

    completion = client.chat.completions.create(
        model=f"{model}",
        messages=messages,
        temperature=0.3,
        stream=True,
        tools=[
            {
                "type": "builtin_function",
                "function": {
                    "name": "$web_search",
                },
            }
        ]
    )
    return completion


def linked_chatbot(model):
    global history, history_index, is_stopped
    original_index = history_index
    finish_reason = None
    final_response = ""
    tool_calls_buffer = []

    try:
        while finish_reason is None or finish_reason == "tool_calls":
            interruption = handle_interruption(original_index)
            if interruption:
                # 如果被中断且有部分回复，保存部分回复和中断消息
                if final_response:
                    append_message({
                        "role": "assistant",
                        "content": final_response
                    }, original_index)
                    append_message({
                        "role": "assistant",
                        "content": "回答已被中断"
                    }, original_index)
                return interruption

            try:
                stream = pre_link(model)
                for chunk in stream:
                    if is_stopped:
                        # 如果被中断且有部分回复，保存部分回复和中断消息
                        if final_response:
                            append_message({
                                "role": "assistant",
                                "content": final_response
                            }, original_index)
                            append_message({
                                "role": "assistant",
                                "content": "回答已被中断"
                            }, original_index)
                        return

                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason
                    
                    # 处理工具调用
                    if choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            if tool_call.index >= len(tool_calls_buffer):
                                tool_calls_buffer.append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""},
                                    "type": ""
                                })
                            
                            if tool_call.id:
                                tool_calls_buffer[tool_call.index]["id"] = tool_call.id
                            if tool_call.function.name:
                                tool_calls_buffer[tool_call.index]["function"]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                tool_calls_buffer[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                            if tool_call.type:
                                tool_calls_buffer[tool_call.index]["type"] = tool_call.type
                    
                    # 处理普通消息
                    elif choice.delta.content:
                        yield choice.delta.content
                        final_response += choice.delta.content

            except Exception as e:
                error_msg = handle_api_error(e, original_index)
                yield error_msg
                return

            if finish_reason == "tool_calls":
                interruption = handle_interruption(original_index)
                if interruption:
                    # 如果被中断且有部分回复，保存部分回复和中断消息
                    if final_response:
                        append_message({
                            "role": "assistant",
                            "content": final_response
                        }, original_index)
                        append_message({
                            "role": "assistant",
                            "content": "回答已被中断"
                        }, original_index)
                    return interruption

                # 如果有工具调用，先添加带有tool_calls的消息
                if tool_calls_buffer:
                    message_dict = {
                        "role": "assistant",
                        "content": final_response if final_response else "",
                        "tool_calls": tool_calls_buffer
                    }
                    append_message(message_dict, original_index)

                    # 处理每个工具调用
                    for tool_call in tool_calls_buffer:
                        interruption = handle_interruption(original_index)
                        if interruption:
                            # 如果在处理工具调用时被中断，添加中断消息
                            append_message({
                                "role": "assistant",
                                "content": "回答已被中断"
                            }, original_index)
                            return interruption

                        tool_call_name = tool_call["function"]["name"]
                        tool_call_arguments = json.loads(tool_call["function"]["arguments"])
                        if tool_call_name == "$web_search":
                            tool_result = search_impl(tool_call_arguments)
                        else:
                            tool_result = f"Error: unable to find tool by name '{tool_call_name}'"

                        # 添加工具响应
                        append_message({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call_name,
                            "content": json.dumps(tool_result),
                        }, original_index)

                    # 重置缓冲区
                    tool_calls_buffer = []
                    final_response = ""

            elif finish_reason:
                interruption = handle_interruption(original_index)
                if interruption:
                    # 如果被中断且有部分回复，保存部分回复和中断消息
                    if final_response:
                        append_message({
                            "role": "assistant",
                            "content": final_response
                        }, original_index)
                        append_message({
                            "role": "assistant",
                            "content": "回答已被中断"
                        }, original_index)
                    return interruption

                if final_response:
                    append_message({
                        "role": "assistant",
                        "content": final_response
                    }, original_index)

    except Exception as e:
        print(f"Error in linked_chatbot: {str(e)}")
        error_msg = handle_api_error(e, original_index)
        yield error_msg


@app.route('/chat', methods=['POST'])
def chat():
    global is_stopped, history_index, history
    is_stopped = False
    current_index = history_index

    try:
        data = request.get_json()
        user_input = data.get('input')
        model_choice = data.get('model', 'moonshot-v1-8k')
        search_toggle = data.get('search-toggle', 'no')
        message_index = data.get('message_index')
        print()
        print(f"chat: 收到请求 - input: {user_input}, model: {model_choice}, search: {search_toggle}, message_index: {message_index}")

        # 检查必要的参数
        if not user_input:
            print("chat: 输入为空，返回错误")
            return Response(
                f"data: {json.dumps({'reply': '输入不能为空', 'done': True})}\n\n",
                mimetype='text/event-stream'
            )

        if message_index is not None:
            try:
                message_index = int(message_index)
                # 保存系统消息和之前的对话
                new_history = []
                message_count = -1  # 从-1开始，这样第一个非系统消息的索引是0
                
                for msg in history[current_index]:
                    if msg["role"] == "system":
                        new_history.append(msg)
                    else:
                        message_count += 1
                        if message_count < message_index:
                            new_history.append(msg)
                        elif message_count == message_index:
                            # 替换要编辑的消息
                            new_history.append({"role": "user", "content": user_input})
                
                # 更新历史记录
                history[current_index] = new_history
                print(f"chat: 更新历史记录完成，当前消息数: {len(new_history)}")
            except (ValueError, TypeError) as e:
                print(f"chat: 消息索引处理错误 - {str(e)}")
                return Response(
                    f"data: {json.dumps({'reply': '消息索引无效', 'done': True})}\n\n",
                    mimetype='text/event-stream'
                )
        else:
            # 如果不是编辑操作，直接添加新消息
            append_message({"role": "user", "content": user_input}, current_index)
            print("chat: 添加新消息完成")

        def generate():
            global is_stopped
            response_text = ""
            try:
                print("generate: 开始生成回复")
                for chunk in chatbot(user_input, model_choice, search_toggle, message_index):
                    if chunk:
                        response_text += chunk
                        yield f"data: {json.dumps({'reply': chunk, 'done': False})}\n\n"
                    
                    if is_stopped:
                        if response_text:
                            append_message({"role": "assistant", "content": response_text}, current_index)
                        append_message({"role": "assistant", "content": "回答已被中断"}, current_index)
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        print("generate: 生成被中断")
                        return

                yield f"data: {json.dumps({'done': True})}\n\n"
                print("generate: 生成完成")
            except Exception as e:
                print(f"generate: 生成错误 - {str(e)}")
                error_msg = "发生错误，请稍后重试"
                append_message({"role": "assistant", "content": error_msg}, current_index)
                yield f"data: {json.dumps({'reply': error_msg, 'done': True})}\n\n"
            finally:
                is_stopped = False
                print("generate: 函数结束")

        print("chat: 返回响应流")
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        print(f"chat: 路由错误 - {str(e)}")
        return Response(
            f"data: {json.dumps({'reply': '服务器错误，请稍后重试', 'done': True})}\n\n",
            mimetype='text/event-stream'
        )


@app.route('/set_stop', methods=['POST'])
def set_stop():
    global is_stopped
    data = request.get_json()  # 获取请求的JSON数据
    is_stopped = data.get("input1") == "True"  # 根据输入设置是否停止
    return jsonify({'status': 'success', 'stopped': is_stopped})  # 返回成功状态和当前停止状态


@app.route('/')
def index():
    return render_template('index.html')  # 返回主页的HTML模板


# 设置上传文件的目录
UPLOAD_FOLDER = r"D:\桌面\upload_files"  # 设置文件上传目录路径
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 配置上传目录

# 设置允许的文件类型
ALLOWED_EXTENSIONS = {'.txt', '.jpg', '.jpeg', '.png', '.pdf', '.docx', '.xlsx', '.pptx', '.doc',
                      '.xlsm'}  # 定义允许上传的文件类型


# 检查文件扩展名是否合法
def allowed_file(filename):
    if not filename:  # 如果没有文件名，返回False
        return False
    # 处理以点号开头的文件名
    if filename.startswith('.'):
        ext = filename.lower()  # 整个文件名就是扩展名
        return ext in ALLOWED_EXTENSIONS  # 检查扩展名是否在允许的文件类型中
    # 处理正常的文件名
    ext = os.path.splitext(filename)[1].lower()  # 获取文件扩展名并转为小写
    return ext in ALLOWED_EXTENSIONS  # 检查扩展名是否在允许的文件类型中


# 文件上传路由
@app.route('/upload', methods=['POST'])
def upload_file():
    # 修改全局变量
    global filespathes, add_files, length, history_index

    # 获取当前对话的文件数量
    length = len(filespathes[history_index])  # 修改为当前对话的所有文件数量

    # 检查请求中是否包含文件
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part"}), 400  # 如果没有文件部分，返回错误

    files = request.files.getlist('files[]')  # 获取用户上传的文件列表

    # 记录上传的文件数量
    add_files = len(files)

    # 如果用户没有选择文件，返回错误
    if len(files) == 0:
        return jsonify({"error": "No selected files"}), 400

    uploaded_files = []  # 用于存储上传成功的文件信息
    for index, file in enumerate(files):
        original_filename = file.filename  # 获取文件的原始文件名
        # 处理空文件名或只有扩展名的情况
        if not original_filename or original_filename.strip() == '' or original_filename.lstrip('.').strip() == '':
            # 生成一个默认的文件名，使用时间戳和索引确保唯一性
            base_filename = f"unnamed_file_{int(time.time())}_{index}"
            # 如果文件有扩展名，保留它
            if '.' in original_filename:
                ext = original_filename.rsplit('.', 1)[1].lower()
                filename = f"{base_filename}.{ext}"
            else:
                filename = f"{base_filename}.txt"  # 默认使用.txt扩展名
        else:
            filename = original_filename  # 保持原始文件名

        # 如果文件类型允许，保存文件
        if allowed_file(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 设置文件保存路径

            # 保存文件到服务器
            file.save(file_path)

            # 获取文件的绝对路径
            absolute_file_path = os.path.abspath(file_path)

            # 保存文件的绝对路径到全局变量
            filespathes[history_index].append(absolute_file_path)

            # 将上传文件的信息添加到上传成功的文件列表中
            uploaded_files.append({
                'filename': filename,
                'absolute_path': absolute_file_path,
                'mimetype': file.mimetype,
                'size': os.path.getsize(file_path)  # 获取文件大小
            })
        else:
            return jsonify({"error": f"File {filename} type not allowed"}), 400  # 文件类型不被允许，返回错误

    # 返回上传成功的消息和文件信息（包括绝对路径）
    return jsonify({
        'message': 'Files uploaded successfully',
        'files': uploaded_files
    }), 200  # 返回成功消息和文件信息


# 设置对话历史的索引
@app.route('/set_index', methods=['POST'])
def set_index():
    global history, history_index, filespathes, length, add_files, is_stopped, client

    data = request.get_json()  # 获取请求的数据
    new_index = data.get("input1", 100)  # 获取新的索引
    print("现在切换到的对话索引是：", new_index)
    old_index = history_index  # 保存旧的索引

    # 如果当前对话有未完成的请求（即最后一条是用户消息），添加中断消息
    if old_index < len(history) and len(history[old_index]) > 0:
        last_message = history[old_index][-1]
        # 只有在以下条件都满足时才添加中断消息：
        # 1. 最后一条消息是用户消息
        # 2. 当前不是在同一个对话内（防止重复添加）
        # 3. 当前对话没有被标记为已中断
        if (last_message.get("role") == "user" and 
            old_index != new_index and 
            not any(msg.get("content") == "回答已被中断" for msg in history[old_index] if msg.get("role") == "assistant")):
            
            history[old_index].append({
                "role": "assistant",
                "content": "回答已被中断"
            })

    # 设置当前对话的历史索引
    history_index = new_index

    add_files = 0  # 初始化添加的文件数

    # 如果当前索引对应的对话历史不存在，创建一个新的对话历史
    if len(history) == history_index:
        history.append([])
        # 为新对话添加系统提示信息
        history[history_index].append({"role": "system",
                                       "content": start_message})

    # 如果当前索引对应的文件路径列表不存在，创建一个新的文件路径列表
    if len(filespathes) == history_index:
        filespathes.append([])

    length = len(history[history_index])  # 更新当前对话的消息数量

    print(history[history_index])  # 输出当前对话历史
    print()

    return '', 204  # 返回空响应


# 添加函数，用于生成对话摘要，默认使用deepseek-chat模型
def generate_conversation_summary(conversation: List[Dict]) -> str:
    # 将对话内容整理成一个字符串
    conversation_text = ""
    for msg in conversation:
        if msg.get("role") == "user":
            conversation_text += f"用户: {msg['content']}\n"
        elif msg.get("role") == "assistant":
            conversation_text += f"助手: {msg['content']}\n"

    client = deepseek_client        

    # 使用 AI 生成摘要
    try:
        summary_messages = [
            {"role": "system",
             "content": "请为以下对话生成一个简短的标题（不超过20个字），这个标题应该概括对话的主要内容。"},
            {"role": "user", "content": conversation_text}
        ]

        # 请求生成摘要
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=summary_messages,
            temperature=0.3,  # 设置生成文本的温度，控制输出的多样性
        )

        # 获取生成的摘要
        summary = completion.choices[0].message.content.strip()
        return summary  # 返回摘要
    except Exception as e:
        print(f"生成摘要时出错: {str(e)}")
        return f"对话 {len(history)}"  # 如果出错，返回对话的默认标题


# 生成对话摘要的路由
@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()  # 获取请求数据
        conversation = data.get('conversation', [])  # 获取对话内容

        # 构建对话文本
        conversation_text = ""
        for msg in conversation:
            if 'user' in msg:
                conversation_text += f"用户: {msg['user']}\n"
            if 'bot' in msg:
                conversation_text += f"助手: {msg['bot']}\n"

        # 使用 AI 生成摘要
        summary_messages = [
            {"role": "system",
             "content": "请为以下对话生成一个简短的标题（不超过20个字），这个标题应该概括对话的主要内容。"},
            {"role": "user", "content": conversation_text}
        ]

        client = deepseek_client

        # 请求生成摘要
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=summary_messages,
            temperature=0.3,
        )

        # 获取并返回生成的摘要
        summary = completion.choices[0].message.content.strip()
        return jsonify({'summary': summary})
    except Exception as e:
        print(f"生成摘要时出错: {str(e)}")
        return jsonify({'summary': None})  # 返回None表示生成失败


# 删除对话的路由
@app.route('/delete_conversation', methods=['POST'])
def delete_conversation():
    try:
        global history, filespathes, history_index
        data = request.get_json()  # 获取请求数据
        index = data.get('index')  # 获取要删除的对话索引

        # 如果没有提供索引，返回错误
        if index is None:
            return jsonify({'status': 'error', 'message': 'No index provided'})

        # 如果索引有效，删除对应的对话和文件路径
        if 0 <= index < len(history):
            # 删除对话历史和对应的文件路径
            history.pop(index)
            if index < len(filespathes):  # 确保filespathes中有对应索引
                filespathes.pop(index)

            # 如果删除的是当前对话或之前的对话，需要调整history_index
            if history_index >= len(history):
                history_index = max(0, len(history) - 1)

            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid index'})
    except Exception as e:
        print(f"Error deleting conversation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})  # 返回错误信息


# 清除文件路径的路由
@app.route('/clear_files', methods=['POST'])
def clear_files():
    global filespathes, history_index, length, add_files
    
    # 清空当前对话的文件路径列表
    if history_index < len(filespathes):
        filespathes[history_index] = []
    
    # 重置文件相关的计数器
    length = 0
    add_files = 0
    
    return jsonify({'status': 'success'})


# 获取所有用户数据的路由
@app.route('/users', methods=['GET'])
def get_all_users():
    conn = None
    try:
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        c.execute('SELECT id, username, email FROM users')  # 注意不返回密码
        users = c.fetchall()
        
        # 将结果转换为字典列表
        users_list = [{'id': user[0], 'username': user[1], 'email': user[2]} for user in users]
        
        print("\n当前所有用户：")
        for user in users_list:
            print(f"ID: {user['id']}, 用户名: {user['username']}, 邮箱: {user['email']}")
        
        return jsonify(users_list), 200
    except Exception as e:
        print(f"\n获取用户列表失败 - 原因：{str(e)}")
        return jsonify({'message': '获取用户列表失败'}), 500
    finally:
        if conn:
            conn.close()


# 删除用户的路由
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    conn = None
    try:
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        
        # 先检查用户是否存在
        c.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        
        if user is None:
            print(f"\n删除失败 - 原因：用户ID {user_id} 不存在")
            return jsonify({'message': '用户不存在'}), 404
            
        # 删除用户
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        
        print(f"\n成功删除用户！")
        print(f"用户名: {user[0]}")
        print(f"邮箱: {user[1]}\n")
        
        return jsonify({'message': '用户删除成功'}), 200
    except Exception as e:
        print(f"\n删除用户失败 - 原因：{str(e)}")
        if conn:
            conn.rollback()
        return jsonify({'message': '删除用户失败'}), 500
    finally:
        if conn:
            conn.close()


@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    conn = None
    try:
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        
        # 获取用户信息（不包括密码）
        c.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        
        if user is None:
            print(f"\n获取用户信息失败 - 原因：用户ID {user_id} 不存在")
            return jsonify({'message': '用户不存在'}), 404
            
        print(f"\n成功获取用户信息！")
        print(f"用户名: {user[0]}")
        print(f"邮箱: {user[1]}\n")
        
        return jsonify({
            'username': user[0],
            'email': user[1]
        }), 200
    except Exception as e:
        print(f"\n获取用户信息失败 - 原因：{str(e)}")
        return jsonify({'message': '获取用户信息失败'}), 500
    finally:
        if conn:
            conn.close()

# 发送重置密码验证码的路由
@app.route('/send_reset_code', methods=['POST'])
def send_reset_code():
    conn = None
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        
        if not all([username, email]):
            return jsonify({'message': '用户名和邮箱都是必填的'}), 400
            
        # 验证用户名和邮箱是否匹配
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND email = ?', (username, email))
        user = c.fetchone()
        
        if not user:
            return jsonify({'message': '用户名或邮箱不正确'}), 400
            
        # 生成验证码
        code = generate_verification_code()
        
        # 发送验证码邮件
        if send_verification_email(email, code):
            # 存储重置密码的验证码
            verification_codes[email] = (code, time.time())
            return jsonify({'message': '验证码已发送'}), 200
        else:
            return jsonify({'message': '验证码发送失败，请稍后重试'}), 500
            
    except Exception as e:
        print(f"发送重置密码验证码失败: {str(e)}")
        return jsonify({'message': '发送验证码失败，请稍后重试'}), 500
    finally:
        if conn:
            conn.close()

def generate_random_password():
    """
    生成一个随机密码，包含大小写字母、数字和特殊字符
    """
    import string
    import random
    
    # 定义字符集
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = "!@#$%^&*"
    
    # 确保密码包含每种字符
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special)
    ]
    
    # 再随机添加4个字符，确保密码长度为8
    all_characters = lowercase + uppercase + digits + special
    password.extend(random.choice(all_characters) for _ in range(4))
    
    # 打乱密码字符顺序
    random.shuffle(password)
    
    return ''.join(password)

# 验证重置密码的路由
@app.route('/verify_reset_code', methods=['POST'])
def verify_reset_code():
    conn = None
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        code = data.get('code')
        
        if not all([username, email, code]):
            return jsonify({'message': '所有字段都是必填的'}), 400
            
        # 验证用户名和邮箱是否匹配
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ? AND email = ?', (username, email))
        user = c.fetchone()
        
        if not user:
            return jsonify({'message': '用户名或邮箱不正确'}), 400
            
        # 验证验证码
        if email not in verification_codes:
            return jsonify({'message': '请先获取验证码'}), 400
            
        stored_code, timestamp = verification_codes[email]
        if time.time() - timestamp > VERIFICATION_CODE_EXPIRY:
            # 验证码过期，删除记录
            del verification_codes[email]
            return jsonify({'message': '验证码已过期，请重新获取'}), 400
            
        if code != stored_code:
            return jsonify({'message': '验证码错误'}), 400
            
        # 生成新的随机密码
        new_password = generate_random_password()
        
        # 更新数据库中的密码
        hashed_password = generate_password_hash(new_password)
        c.execute('UPDATE users SET password = ? WHERE id = ?', (hashed_password, user[0]))
        conn.commit()
        
        # 验证通过，返回新密码
        return jsonify({
            'message': '验证成功，已重置密码',
            'password': new_password  # 返回明文密码
        }), 200
            
    except Exception as e:
        print(f"验证重置密码验证码失败: {str(e)}")
        return jsonify({'message': '验证失败，请稍后重试'}), 500
    finally:
        if conn:
            conn.close()

# 设置头像上传目录
STATIC_FOLDER = 'static'
AVATARS_FOLDER = 'avatars'
AVATAR_FOLDER = os.path.join(STATIC_FOLDER, AVATARS_FOLDER)
AVATAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), AVATAR_FOLDER)

# 确保头像目录存在
if not os.path.exists(AVATAR_PATH):
    os.makedirs(AVATAR_PATH)

# 确保默认头像存在
DEFAULT_AVATAR = 'default_avatar.png'
DEFAULT_AVATAR_PATH = os.path.join(AVATAR_PATH, DEFAULT_AVATAR)
if not os.path.exists(DEFAULT_AVATAR_PATH):
    # 创建一个简单的默认头像（可以替换为你的默认头像文件）
    img = Image.new('RGB', (100, 100), color='#cccccc')
    d = ImageDraw.Draw(img)
    img.save(DEFAULT_AVATAR_PATH)

app.config['AVATAR_FOLDER'] = AVATAR_FOLDER

# 允许的图片类型
ALLOWED_AVATAR_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_avatar_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AVATAR_EXTENSIONS

# 上传头像的路由
@app.route('/upload_avatar', methods=['POST'])
@login_required
def upload_avatar():
    try:
        if 'avatar' not in request.files:
            return jsonify({'message': '没有选择文件'}), 400
            
        file = request.files['avatar']
        if file.filename == '':
            return jsonify({'message': '没有选择文件'}), 400
            
        if file and allowed_avatar_file(file.filename):
            # 生成安全的文件名（使用用户ID作为文件名前缀）
            ext = os.path.splitext(file.filename)[1]
            filename = secure_filename(f"user_{session['user_id']}{ext}")
            filepath = os.path.join(AVATAR_PATH, filename)
            
            # 保存文件
            file.save(filepath)
            
            try:
                # 更新数据库中的头像路径
                conn = sqlite3.connect('users.db', timeout=20)
                c = conn.cursor()
                
                # 获取旧的头像路径
                c.execute('SELECT avatar_path FROM users WHERE id = ?', (session['user_id'],))
                old_avatar = c.fetchone()
                
                # 如果存在旧头像且不是默认头像，删除它
                if old_avatar and old_avatar[0] and old_avatar[0] != DEFAULT_AVATAR:
                    old_avatar_path = os.path.join(AVATAR_PATH, old_avatar[0])
                    if os.path.exists(old_avatar_path):
                        os.remove(old_avatar_path)
                
                # 更新数据库
                c.execute('UPDATE users SET avatar_path = ? WHERE id = ?', 
                         (filename, session['user_id']))
                conn.commit()
                
                return jsonify({
                    'message': '头像上传成功',
                    'avatar_url': f'/{AVATAR_FOLDER}/{filename}'
                }), 200
            except Exception as e:
                # 如果数据库操作失败，删除已上传的文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise e
            finally:
                if conn:
                    conn.close()
    except Exception as e:
        print(f"上传头像失败: {str(e)}")
        return jsonify({'message': '上传失败，请稍后重试'}), 500

# 获取用户头像的路由
@app.route('/get_avatar/<int:user_id>')
def get_avatar(user_id):
    try:
        conn = sqlite3.connect('users.db', timeout=20)
        c = conn.cursor()
        c.execute('SELECT avatar_path FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        conn.close()
        
        if result and result[0]:
            # 检查文件是否存在
            avatar_path = os.path.join(AVATAR_PATH, result[0])
            if os.path.exists(avatar_path):
                return jsonify({
                    'avatar_url': f'/{AVATAR_FOLDER}/{result[0]}'
                }), 200
            else:
                # 如果文件不存在，清除数据库中的记录并返回默认头像
                conn = sqlite3.connect('users.db', timeout=20)
                c = conn.cursor()
                c.execute('UPDATE users SET avatar_path = ? WHERE id = ?', (DEFAULT_AVATAR, user_id))
                conn.commit()
                conn.close()
                return jsonify({
                    'avatar_url': f'/{AVATAR_FOLDER}/{DEFAULT_AVATAR}'
                }), 200
        else:
            return jsonify({
                'avatar_url': f'/{AVATAR_FOLDER}/{DEFAULT_AVATAR}'
            }), 200
    except Exception as e:
        print(f"获取头像失败: {str(e)}")
        return jsonify({'message': '获取头像失败'}), 500

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)






