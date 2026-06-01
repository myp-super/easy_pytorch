# Git 完全使用指南

> 从零开始，覆盖基础概念、日常使用、离线工作、国内网络问题等全部场景。

---

## 目录

- [1. Git 是什么](#1-git-是什么)
- [2. 安装与首次配置](#2-安装与首次配置)
- [3. 核心概念速览](#3-核心概念速览)
- [4. 基础操作速查](#4-基础操作速查)
- [5. 日常工作流](#5-日常工作流)
- [6. 撤销与回退](#6-撤销与回退)
- [7. 分支与合并](#7-分支与合并)
- [8. 与 GitHub 协作](#8-与-github-协作)
- [9. 离线 / 无网络使用](#9-离线--无网络使用)
- [10. 国内网络问题解决](#10-国内网络问题解决)
- [11. .gitignore 忽略文件](#11-gitignore-忽略文件)
- [12. 常见问题速查](#12-常见问题速查)
- [13. 最佳实践与习惯](#13-最佳实践与习惯)

---

## 1. Git 是什么

**Git** 是一个**分布式版本控制系统**。通俗理解：

- 它是你项目的"存档系统"——每次 `commit` 就像游戏存档，你可以随时回到任意一个存档点
- 它是"时间机器"——你能看到项目在任意时间点的样子
- 它是"协作平台"——多个人同时改代码不会互相覆盖

**为什么需要它？**

| 没有 Git | 有 Git |
|-----------|--------|
| `论文_最终版.docx`、`论文_最终版2.docx`、`论文_真的最终版.docx` | 一个文件，所有历史版本可回溯 |
| 误删代码找不回来 | `git checkout` 一秒恢复 |
| 不知道谁改了哪行、为什么改 | `git log` + `git blame` 清清楚楚 |
| 多人同时改一个文件互相覆盖 | 各自在分支上工作，合并时解决冲突 |

**Git ≠ GitHub**：

- **Git**：版本控制工具，装在你电脑上，可以完全离线使用
- **GitHub**：托管 Git 仓库的网站，用来在云端备份和协作
- 你可以在没有 GitHub 的情况下使用 Git；但通常两者配合使用

---

## 2. 安装与首次配置

### 2.1 安装 Git

**Windows**：去 [git-scm.com](https://git-scm.com) 下载安装包，一路默认选项即可。

安装完成后，在终端里验证：

```bash
git --version
# 输出类似：git version 2.44.0
```

### 2.2 配置用户名和邮箱

每个 commit 都会记录作者信息，先告诉 Git 你是谁：

```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"
```

> `--global` 表示全局配置，对本机所有仓库生效。如果想对单个项目用不同的身份，在那个项目目录下不带 `--global` 运行即可。

**检查配置是否生效：**

```bash
git config --global --list
# 输出：
# user.name=myp-super
# user.email=your-email@example.com
```

### 2.3 配置 SSH 密钥（推荐，解决国内网络问题）

HTTPS 方式在国内经常连不上 GitHub。**强烈建议配置 SSH 密钥**，走 SSH 协议访问 GitHub。

#### 第一步：检查是否已有密钥

```bash
ls ~/.ssh/id_ed25519.pub
# 如果显示 "No such file or directory"，说明还没有，需要生成
# 如果显示了路径，跳到第三步
```

#### 第二步：生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
```

一路回车即可（不需要设置密码短语，除非你有特殊安全需求）。

#### 第三步：复制公钥

**Windows (Git Bash / PowerShell)：**

```bash
cat ~/.ssh/id_ed25519.pub
```

把输出的内容（以 `ssh-ed25519` 开头，以邮箱结尾）全部复制。

#### 第四步：添加到 GitHub

1. 打开 [github.com/settings/keys](https://github.com/settings/keys)
2. 点击 **New SSH Key**
3. Title 随便填（如 "我的电脑"）
4. Key 粘贴刚才复制的内容
5. 点击 **Add SSH Key**

#### 第五步：测试连接

```bash
ssh -T git@github.com
# 看到 "Hi your-username! You've successfully authenticated..." 就成功了
```

---

## 3. 核心概念速览

理解这几个概念，Git 就不再神秘：

```
┌─────────────┐    git add    ┌─────────────┐   git commit   ┌─────────────┐
│  工作区      │ ────────────→ │  暂存区      │ ─────────────→ │  本地仓库    │
│  (Working   │               │  (Staging   │               │  (Local     │
│   Directory) │               │   Area)     │               │   Repo)     │
└─────────────┘               └─────────────┘               └──────┬──────┘
                                                                   │
                                                          git push │  │ git pull
                                                                   │  │ git fetch
                                                                   ↓  ↓
                                                            ┌─────────────┐
                                                            │  远程仓库    │
                                                            │  (GitHub)   │
                                                            └─────────────┘
```

| 概念 | 解释 | 类比 |
|------|------|------|
| **工作区 (Working Directory)** | 你电脑上的文件夹，你在这里编辑代码 | 你的书桌，你在上面写写画画 |
| **暂存区 (Staging Area)** | 临时存放你想保存的修改 | 草稿箱，你挑出要存档的文件放进去 |
| **本地仓库 (Local Repo)** | 存在你电脑上的完整版本历史 | 你的私人档案柜，所有存档都在里面 |
| **远程仓库 (Remote)** | GitHub 上的仓库，用来备份和协作 | 云端备份 + 共享档案柜 |
| **提交 (Commit)** | 一次存档，记录当时的文件快照 | 游戏存档点 |
| **分支 (Branch)** | 独立的开发线，互不影响 | 平行宇宙，你在不同宇宙里做不同的事 |
| **HEAD** | 指向你当前所在的位置（通常是某个分支的最新 commit） | "你现在在哪里"的指针 |
| **标签 (Tag)** | 给某个 commit 起个有意义的名字 | 书签，标记重要节点如 v1.0 |

---

## 4. 基础操作速查

### 4.1 创建仓库

```bash
# 在现有项目目录里初始化
cd 你的项目目录
git init

# 或者从 GitHub 克隆一个已有的仓库
git clone git@github.com:用户名/仓库名.git
```

### 4.2 查看状态

```bash
git status          # 查看哪些文件改了、哪些在暂存区
git status --short  # 简洁输出
```

`git status` 会告诉你：
- 哪些文件被修改了但没暂存（红色）
- 哪些文件已暂存等待提交（绿色）
- 哪些文件没被 Git 跟踪（红色 `??`）

### 4.3 添加文件到暂存区

```bash
git add 文件名          # 添加指定文件
git add 文件1 文件2      # 添加多个文件
git add .               # 添加当前目录下所有变更
git add -A              # 添加整个仓库所有变更（包括删除的文件）
git add -u              # 只添加已跟踪文件的变更（不包括新文件）
```

### 4.4 提交

```bash
git commit -m "提交信息"
```

**写好提交信息**：让别人（和一个月后的你自己）能看懂你做了什么。

```bash
# 好的提交信息
git commit -m "修复登录页面密码框无法输入中文的问题"

# 不好的提交信息
git commit -m "fix"
git commit -m "更新"
```

> 如果不小心写了 `git commit` 没加 `-m`，会进入 vim 编辑器。按 `i` 开始输入，输完后按 `Esc`，再输入 `:wq` 回车退出。

### 4.5 查看历史

```bash
git log                    # 完整历史
git log --oneline          # 一行一个 commit，最常用
git log --oneline -10      # 只看最近 10 条
git log --oneline --graph  # 显示分支图
git log --oneline --all    # 显示所有分支
git log -p                 # 显示每次提交的具体改动
git log --grep="关键词"    # 搜索提交信息
git log --author="用户名"  # 按作者筛选
```

### 4.6 查看改动

```bash
git diff                  # 工作区 vs 暂存区（还没 add 的改动）
git diff --staged         # 暂存区 vs 上次 commit（已经 add 但还没 commit 的改动）
git diff HEAD             # 工作区 vs 上次 commit（所有没 commit 的改动）
git diff 分支1..分支2      # 两个分支之间的差异
git show 提交ID           # 查看某次提交的详细改动
```

---

## 5. 日常工作流

### 5.1 每日标准流程

这是你每天打开电脑开始工作时要做的事：

```bash
# ========== 早上：同步最新代码 ==========
cd 你的项目目录
git pull origin main        # 拉取 GitHub 上最新的代码

# ========== 工作中：不断存档 ==========
# 修改代码...
git add .
git commit -m "完成了 XX 功能的一半"

# 继续修改...
git add .
git commit -m "完成了 XX 功能，测试通过"

# ========== 晚上：推送到 GitHub ==========
git push origin main
```

**核心习惯**：

- **commit 要小、要频繁**：一个小功能完成就 commit 一次，不要攒了一大堆才 commit
- **push 要规律**：每天至少 push 一次，把本地的工作备份到云端
- **pull 在 push 之前**：push 之前先 pull，避免冲突

### 5.2 单个文件的工作流

```bash
# 修改了一个文件
vim README.md

# 查看改了什么
git diff README.md

# 暂存这个文件
git add README.md

# 提交
git commit -m "更新 README：添加环境配置说明"

# 推送
git push origin main
```

### 5.3 一条命令搞定 add + commit

```bash
# 只对已跟踪的文件有效
git commit -am "提交信息"
# 等价于：git add -u && git commit -m "提交信息"
```

> `-a` 会自动 add 已跟踪文件的修改，但**不会添加新文件**。新文件还是需要先 `git add`。

### 5.4 查看当前工作进度

```bash
git status        # 随时查看，了解自己在哪里、改了什么
git log --oneline -5  # 看看最近的提交历史
```

---

## 6. 撤销与回退

### 6.1 撤销工作区修改（还没 git add）

```bash
# 撤销单个文件的修改
git checkout -- 文件名

# 撤销所有修改（慎用！）
git checkout -- .
```

### 6.2 撤销暂存区（已经 git add 但还没 commit）

```bash
# 把文件从暂存区拉回工作区（修改还在，只是取消暂存）
git reset HEAD 文件名

# 全部取消暂存
git reset HEAD
```

### 6.3 修改最近一次 commit

```bash
# 补充遗漏的文件到上次 commit（不改提交信息）
git add 遗漏的文件
git commit --amend --no-edit

# 修改上次 commit 的提交信息
git commit --amend -m "新的提交信息"
```

> ⚠️ 如果已经 push 过，amend 后的 commit 需要用 `git push --force` 才能推上去。**协作分支上不要用 amend。**

### 6.4 撤销 commit（保留改动）

```bash
# 撤销最近 1 次 commit，改动回到暂存区
git reset --soft HEAD~1

# 撤销最近 1 次 commit，改动回到工作区（未暂存）
git reset --mixed HEAD~1

# 撤销最近 N 次 commit
git reset --soft HEAD~3
```

### 6.5 彻底撤销 commit（丢弃改动）

```bash
# 彻底丢弃最近 1 次 commit 的所有改动（不可恢复！）
git reset --hard HEAD~1

# 回到某个特定的 commit，丢弃之后的一切
git reset --hard 提交ID
```

> ⚠️ `git reset --hard` 不可逆，使用前三思。

### 6.6 安全的撤销方式：revert

```bash
# 创建一个新的 commit 来"反做"某次 commit，不修改历史
git revert 提交ID
```

**reset vs revert**：

| | reset | revert |
|------|-------|--------|
| 原理 | 删除 commit | 反向操作，产生新的 commit |
| 历史 | 改写历史 | 保留完整历史 |
| 已 push 后使用 | ❌ 不推荐 | ✅ 安全 |
| 适用场景 | 本地还没 push 的 commit | 任何场景，尤其是协作分支 |

### 6.7 恢复误删的文件

```bash
# 如果文件被 Git 跟踪过，可以从历史中恢复
git checkout HEAD -- 被删除的文件
```

---

## 7. 分支与合并

### 7.1 为什么需要分支

分支让你在不影响主线代码的情况下开发新功能或修复 bug：

```
main ────●────●────●────●────●  (稳定版本)
              \
feature ───────●────●────●  (开发中的新功能，不影响 main)
```

### 7.2 分支操作速查

```bash
# 查看所有分支
git branch -a

# 创建分支
git branch 分支名

# 切换分支
git checkout 分支名
# 或（新版 Git 推荐）
git switch 分支名

# 创建并切换到新分支（一行搞定）
git checkout -b 新分支名
# 或
git switch -c 新分支名

# 删除分支
git branch -d 分支名     # 已合并的分支
git branch -D 分支名     # 强制删除（即使没合并）

# 重命名分支
git branch -m 旧名 新名
```

### 7.3 合并分支

```bash
# 1. 切换到目标分支（通常是 main）
git checkout main

# 2. 合并源分支
git merge 源分支名
```

**合并的三种情况**：

| 情况 | 结果 |
|------|------|
| main 没有新 commit（快进合并） | 直接移动指针，无额外 commit |
| 各自有 commit 但没冲突 | 自动合并，产生一个 merge commit |
| 同一文件的同一行都被改了 | **冲突！**需要手动解决 |

### 7.4 解决冲突

当 `git merge` 提示 `CONFLICT` 时：

```bash
# 1. 查看哪些文件冲突
git status

# 2. 打开冲突文件，会看到类似这样的标记：
<<<<<<< HEAD
这是 main 分支上的代码
=======
这是你分支上的代码
>>>>>>> 你的分支名

# 3. 手动编辑，保留正确的代码，删除标记符号

# 4. 标记冲突已解决
git add 冲突的文件

# 5. 完成合并
git commit -m "解决合并冲突：XX 功能与 main 的冲突"
```

### 7.5 变基（Rebase）

把当前分支的 commit "嫁接到"目标分支最新 commit 之后，保持提交历史是一条直线：

```bash
git checkout feature
git rebase main
```

**merge vs rebase**：
- **merge**：保留完整历史，有合并记录，适合公开分支
- **rebase**：历史干净整洁，适合整理自己的开发分支

> ⚠️ 不要对已经 push 的分支做 rebase。

---

## 8. 与 GitHub 协作

### 8.1 关联远程仓库

```bash
# 查看当前的远程仓库
git remote -v

# 添加远程仓库
git remote add origin git@github.com:用户名/仓库名.git

# 修改远程仓库地址
git remote set-url origin git@github.com:用户名/仓库名.git

# 删除远程仓库关联
git remote remove origin
```

### 8.2 推送与拉取

```bash
# 推送当前分支到远程
git push origin 分支名

# 推送并设置上游（之后只需要 git push）
git push -u origin 分支名

# 强制推送（覆盖远程，慎用！）
git push --force origin 分支名

# 从远程拉取最新代码并合并
git pull origin 分支名

# 只拉取（不自动合并）
git fetch origin
```

### 8.3 首次推送本地仓库到 GitHub

```bash
# 1. 在 GitHub 上创建空仓库（不勾选 README、.gitignore）

# 2. 本地操作
cd 你的项目目录
git init
git add -A
git commit -m "首次提交"

# 3. 关联远程
git remote add origin git@github.com:用户名/仓库名.git

# 4. 推送
git branch -M main           # 确保分支名是 main
git push -u origin main
```

### 8.4 Clone vs Pull

| 命令 | 用途 | 什么时候用 |
|------|------|-----------|
| `git clone` | 把远程仓库完整下载到本地 | 第一次获取这个项目 |
| `git pull` | 拉取远程更新并合并到当前分支 | 项目已经在本地，只是同步最新代码 |

### 8.5 Fork 与 Pull Request（参与开源项目）

```
1. Fork：在 GitHub 网页上点 Fork 按钮，把别人的仓库复制到你的账号下
2. Clone：git clone git@github.com:你的用户名/项目.git
3. 修改：创建分支 → 改代码 → commit → push
4. Pull Request：在 GitHub 网页上发起 PR，请求原作者合并你的修改
```

---

## 9. 离线 / 无网络使用

Git 是**分布式**版本控制系统，所有版本历史都存在你的电脑上。**绝大多数操作不需要网络**。

### 9.1 离线可以做的（完全正常）

```bash
git init              # 创建新仓库
git add               # 暂存修改
git commit            # 提交（创建存档）
git log               # 查看历史
git diff              # 查看改动
git branch            # 创建/切换分支
git merge             # 合并分支
git reset             # 回退版本
git checkout -- 文件  # 恢复文件
git stash             # 暂存当前工作现场
git tag               # 打标签
```

**结论：日常工作 90% 的操作，离线都能做。**

### 9.2 离线不能做的（需要网络）

```bash
git push              # 推送到 GitHub → 需要网络
git pull / git fetch  # 从 GitHub 拉取 → 需要网络
git clone             # 从 GitHub 克隆 → 需要网络
```

### 9.3 离线工作策略

当你处于无网络环境（如在路上、在教室、网络故障）时：

```bash
# 正常写代码、正常 commit，没有任何限制
git add -A
git commit -m "完成了第三章的练习"

# 继续工作...
git add .
git commit -m "开始写第四章"

# 继续工作...
git commit -am "第四章完成一半"
```

**本地可以积累任意多个 commit，等有网络了再一次性 push。**

### 9.4 恢复网络后

```bash
# 1. 先拉取远程最新的代码
git pull origin main

# 2. 如果有冲突，解决冲突
# （解决后 git add + git commit）

# 3. 把你离线期间积累的 commit 一次性推上去
git push origin main
```

### 9.5 完整的离线工作策略

```bash
# ===== 离线前（有网时）的准备 =====
git pull origin main          # 确保本地是最新的
# 这样即使之后断网，你也是从最新版本开始工作

# ===== 离线期间 =====
# 正常：写代码 → git add → git commit
# 可以创建分支做实验
git checkout -b experimental
# 各种 commit...
# 实验满意了就合并回去
git checkout main
git merge experimental
# 全部离线完成

# ===== 恢复网络后 =====
git pull origin main          # 同步远程最新代码
git push origin main          # 把离线期间的成果推上去
```

### 9.6 用 Git 做纯本地版本控制

Git 完全可以在没有 GitHub 的情况下使用：

```bash
# 在任何文件夹里
git init
git add -A
git commit -m "v1.0"

# 继续工作...
git add -A
git commit -m "v2.0"

# 随时可以回到 v1.0
git log --oneline   # 找到 v1.0 的 commit ID
git checkout 提交ID -- 文件名   # 恢复某个文件到 v1.0 的状态
```

---

## 10. 国内网络问题解决

### 10.1 问题表现

```bash
# HTTPS 方式经常出现：
fatal: unable to access 'https://github.com/...': 
  Failed to connect to github.com port 443: Connection timed out

# 或者
fatal: unable to access 'https://github.com/...': 
  OpenSSL SSL_read: Connection was reset
```

### 10.2 推荐方案：使用 SSH

这是最稳定、最推荐的方案：

```bash
# 1. 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "你的邮箱"

# 2. 把公钥添加到 GitHub（见第 2.3 节）

# 3. 将现有仓库从 HTTPS 切换到 SSH
git remote set-url origin git@github.com:用户名/仓库名.git

# 4. 验证
git remote -v
# 应该显示：origin  git@github.com:用户名/仓库名.git
```

**为什么 SSH 更稳定？** SSH 走 22 端口，在国内通常比 HTTPS（443 端口）连接 GitHub 更稳定。

### 10.3 方案二：配置代理

如果你有 VPN 或代理软件：

```bash
# 设置代理（端口号根据你的代理软件调整）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 查看代理配置
git config --global --get http.proxy

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

常见代理软件的默认端口：

| 代理软件 | 默认 HTTP 端口 |
|----------|---------------|
| Clash / Clash Verge | 7890 |
| V2Ray / V2RayN | 10809 |
| Shadowsocks | 1080 |
| Trojan | 1080 |

### 10.4 方案三：使用镜像站

```bash
# 使用国内镜像加速 clone（但不推荐长期使用）
git clone https://hub.fastgit.xyz/用户名/仓库名.git
# 然后改回官方地址
git remote set-url origin git@github.com:用户名/仓库名.git
```

### 10.5 HTTPS 和 SSH 之间切换

```bash
# 查看当前用的是哪个
git remote -v

# 切换到 SSH（推荐）
git remote set-url origin git@github.com:用户名/仓库名.git

# 切换回 HTTPS
git remote set-url origin https://github.com/用户名/仓库名.git
```

---

## 11. .gitignore 忽略文件

哪些文件不应该提交到 Git？

- 编译产物（如 Python 的 `__pycache__/`、`.pyc`）
- 依赖包（如 `node_modules/`）
- 环境配置文件（如 `.env`，包含密钥）
- 系统文件（如 `.DS_Store`、`Thumbs.db`）
- IDE 配置（如 `.vscode/`、`.idea/`）
- 大文件（如模型权重 `.pth`、数据集 `.zip`）

在项目根目录创建 `.gitignore` 文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/

# 环境变量（可能包含密钥）
.env
.env.local

# 系统文件
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# 大型文件
*.pth
*.ckpt
*.onnx
*.zip
*.tar.gz
datasets/

# Jupyter
.ipynb_checkpoints/
```

> GitHub 提供了各语言的 `.gitignore` 模板：[github.com/github/gitignore](https://github.com/github/gitignore)

---

## 12. 常见问题速查

### 12.1 我 commit 了不该提交的文件（如大文件、密钥）

```bash
# 从 Git 跟踪中移除，但保留本地文件
git rm --cached 文件名
git commit -m "从版本控制中移除 XX 文件"

# 然后把它加入 .gitignore，防止再次提交
echo "文件名" >> .gitignore
```

### 12.2 提交信息写错了

```bash
# 修改最近一次 commit 的信息
git commit --amend -m "正确的提交信息"

# 如果已经 push 了
git push --force origin 分支名
```

### 12.3 忘记创建分支，直接在 main 上开发了

```bash
# 1. 创建新分支，保存当前工作
git checkout -b feature-xxx

# 2. 切回 main
git checkout main

# 3. 把 main 回退到开发前的状态
git reset --hard origin/main   # 或者 git reset --hard 提交ID
```

### 12.4 git pull 报 "Please specify which branch"

```bash
# 设置当前分支跟踪哪个远程分支
git branch --set-upstream-to=origin/main main
# 之后再 git pull 就可以了
```

### 12.5 合并冲突了想放弃

```bash
git merge --abort    # 回到合并前的状态，什么都没发生
```

### 12.6 想把多个零碎的 commit 合并成一个

```bash
# 合并最近 3 个 commit
git rebase -i HEAD~3

# 在编辑器中，除第一个外，把后面的 pick 改成 squash（或 s）
# 保存退出，编辑新的 commit message
```

### 12.7 切分支时忘了 commit，工作丢了

```bash
# Git 通常会阻止你切换分支。如果真的丢失了：
git reflog                          # 查看所有 HEAD 移动记录
git checkout 找到的提交ID             # 恢复
```

### 12.8 git push --force 之后后悔了

```bash
git reflog                          # 找到被覆盖前的 commit ID
git reset --hard 那个commitID        # 回到被覆盖前的状态
git push --force origin main         # 再次 force push 恢复
```

> 只要 commit 过，Git 几乎永远不会真正丢失数据。`git reflog` 是你的救命稻草。

---

## 13. 最佳实践与习惯

### 13.1 提交习惯

```bash
# ✅ 小步提交，频繁提交
每个逻辑完成点提交一次
commit message 写清楚"做了什么"和"为什么"

# ✅ 提交信息格式
"修复 XX 问题，原因是 YY"
"添加 XX 功能，支持 YY 场景"
"重构 XX 模块，减少重复代码"

# ❌ 避免
一次性提交几百行的改动
"fix"、"update"、"修改" 这种无意义的信息
```

### 13.2 分支策略（简单版）

```
main ──────────●──────────●──────────●──────  （生产稳定版本）
                \        / \        /
develop ────────●──●──●───●──●──●───●──────  （开发主线）
                  \    /
feature-xxx ──────●──●  （独立功能开发，完成后合并到 develop）
```

对于个人项目，简化版：

```
main ────●────●────────●────  （稳定 + 开发都在这里，够用）
           \
test ───────●────●  （偶尔想尝试新东西时再开分支）
```

### 13.3 每日清单

| 时机 | 做什么 |
|------|--------|
| 开始工作 | `git pull` 同步最新代码 |
| 工作中每完成一个小目标 | `git add` + `git commit` |
| 结束工作 | `git push` 推到 GitHub 备份 |
| 随时 | `git status` 了解当前状态 |
| 定期 | `git log --oneline` 回顾工作进度 |

### 13.4 心理安全

> **只要 commit 过，数据就不会真正丢失。** Git 的设计原则就是保护数据。即使你 `reset --hard`、`rebase -i`、甚至删除了分支，都可以通过 `git reflog` 找回。

### 13.5 推荐学习路径

1. **第一天**：学会 `init、add、commit、status、log、diff`——这些就够了
2. **第一周**：学会 `branch、checkout、merge`——处理分支
3. **第二周**：学会 `reset、revert、reflog`——处理"事故"
4. **之后**：碰到问题查文档，慢慢积累

**你不需要一次性学完所有命令。** 90% 的时间你只用到 10 个命令。

---

## 附录：命令速查表

| 命令 | 作用 |
|------|------|
| `git init` | 在当前目录创建仓库 |
| `git clone <url>` | 克隆远程仓库 |
| `git status` | 查看状态 |
| `git add <file>` | 暂存文件 |
| `git add -A` | 暂存所有变更 |
| `git commit -m "msg"` | 提交 |
| `git commit -am "msg"` | 暂存+提交已跟踪文件 |
| `git push origin <branch>` | 推送到远程 |
| `git pull origin <branch>` | 从远程拉取并合并 |
| `git fetch origin` | 从远程拉取（不合并） |
| `git log --oneline` | 查看历史 |
| `git diff` | 查看未暂存的改动 |
| `git diff --staged` | 查看已暂存的改动 |
| `git branch` | 查看分支列表 |
| `git branch <name>` | 创建分支 |
| `git checkout <name>` | 切换分支 |
| `git checkout -b <name>` | 创建并切换分支 |
| `git merge <branch>` | 合并分支 |
| `git merge --abort` | 取消合并 |
| `git reset HEAD <file>` | 取消暂存 |
| `git reset --soft HEAD~1` | 撤销 commit（保留改动） |
| `git reset --hard HEAD~1` | 丢弃最近一次 commit |
| `git checkout -- <file>` | 恢复文件 |
| `git revert <commit>` | 安全撤销某次 commit |
| `git stash` | 暂存当前工作现场 |
| `git stash pop` | 恢复暂存的工作 |
| `git reflog` | 查看所有 HEAD 移动记录 |
| `git remote -v` | 查看远程仓库地址 |
| `git rm --cached <file>` | 停止跟踪文件 |
| `git tag v1.0` | 打标签 |
