# 大文件说明

由于GitHub对文件大小有限制，以下文件需要手动上传：

## 需要手动上传的文件

1. `app.py` - 主应用程序文件 (约58KB)
2. `templates/index.html` - 前端HTML模板 (约179KB)
3. `static/images/default.png` - 默认图片 (约708B)
4. `static/images/deepseek.png` - DeepSeek模型图标 (约16KB)
5. `static/images/kimi.png` - Kimi模型图标 (约11KB)
6. `static/avatars/default_avatar.png` - 默认头像 (约287B)

## 手动上传步骤

1. 克隆仓库到本地
   ```
   git clone https://github.com/theWDY/flask-ai-chat.git
   cd flask-ai-chat
   ```

2. 将上述文件复制到对应目录

3. 提交并推送更改
   ```
   git add .
   git commit -m "添加大文件"
   git push
   ```

## 注意事项

- 确保上传的文件与原始文件内容一致
- 如果文件太大，可以考虑使用Git LFS (Large File Storage)
- 对于二进制文件，请确保使用二进制模式传输