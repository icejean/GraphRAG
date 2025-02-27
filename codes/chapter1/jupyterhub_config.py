c.Authenticator.whitelist = {'ubuntu'}   # 允许使用Jupyter Hub的用户列表，逗号分隔。
c.Authenticator.admin_users = {'ubuntu'}  #Jupyter Hub的管理员用户列表
c.Spawner.notebook_dir = '/home/{username}'  #浏览器登录后进入用户的主目录
c.Spawner.default_url = '/lab'    # 使用Jupyter Lab而不是Notebook
c.JupyterHub.extra_log_file = '/var/log/jupyterhub.log'  #Jupyter Hub额外的日志文件
