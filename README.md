# Flask AI 聊天应用

这是一个基于Flask的AI聊天应用，集成了Moonshot和DeepSeek API，提供了丰富的聊天功能。

## 功能特点

### 主要功能
1. 普通对话 - 与AI进行基础的文本对话
2. 含文件上传的对话 - 支持上传文件并基于文件内容进行对话
3. 含搜索功能的对话 - AI可以联网搜索信息回答问题
4. 含文件与搜索功能的对话 - 结合文件上传和搜索功能

### 附加功能
1. 引用功能 - 为模型的答复添加引用，方便用户查看引用内容和针对引用内容进行对话
2. 对话摘要 - 自动生成对话摘要，也支持手动修改
3. 历史管理 - 可以删除历史对话记录
4. 文件管理 - 可以删除已上传的文件
5. 编辑功能 - 可以修改原来的提问，再次发送给模型

## 安装与配置

### 环境要求
- Python 3.8+
- Flask
- OpenAI Python库

### 安装步骤
1. 克隆仓库
   ```
   git clone https://github.com/theWDY/flask-ai-chat.git
   cd flask-ai-chat
   ```

2. 安装依赖
   ```
   pip install -r requirements.txt
   ```

3. 设置环境变量
   ```
   # Moonshot API密钥
   export MOONSHOT_DEMO_API_KEY=your_moonshot_api_key
   
   # DeepSeek API密钥
   export DEEPSEEK_API_KEY=your_deepseek_api_key
   
   # 邮箱配置（用于验证码发送）
   export SMTP_USER=your_email@qq.com
   export SMTP_PASSWORD=your_email_authorization_code
   ```

4. 运行应用
   ```
   python app.py
   ```

## 使用说明

1. 注册/登录 - 使用邮箱注册账号并登录
2. 选择模型 - 可选择Moonshot或DeepSeek的不同模型
3. 开始对话 - 输入问题并获取AI回答
4. 上传文件 - 点击上传按钮添加文件到对话
5. 启用搜索 - 开启搜索功能让AI联网查询信息

## 注意事项

- 需要有效的Moonshot和DeepSeek API密钥
- 邮箱验证功能需要配置QQ邮箱SMTP服务
- 上传文件目录需要有写入权限

## 许可证

MIT