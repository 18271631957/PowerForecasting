import logging


# 自定义格式化器：自动把 args namespace 按逗号换行
class PrettyArgsFormatter(logging.Formatter):
    def format(self, record):
        # 先执行原始格式化
        log_msg = super().format(record)
        # 如果是 args 配置行，自动美化换行 + 缩进
        if 'args -> namespace(' in log_msg:
            # 把 ", " 替换成 逗号 + 换行 + 缩进
            log_msg = log_msg.replace(", ", ",\n    ")
        return log_msg


def get_logger(process_floder_path, name):
    logger = logging.getLogger(name)

    # 每次获取 logger 时，清空之前绑定的所有 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    filename = f'{process_floder_path}/{name}.log'

    fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')

    # 🔥 关键：使用我们自定义的美化 formatter
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    formatter = PrettyArgsFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
