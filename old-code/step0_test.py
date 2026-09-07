import os
from dotenv import load_dotenv
from openai import OpenAI

print("====程序启动====")
try:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print(f"读取到key：{api_key is not None}")

    # 阿里云百炼OpenAI兼容地址，不需要WorkspaceId
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 任务1：直接文本总结
    resp1 = client.chat.completions.create(
        model="qwen3.8-flash",
        messages=[
            {"role":"system","content":"你是文本总结助手，简短总结内容"},
            {"role":"user","content":"财务管理中流动比率等于流动资产除以流动负债，用来衡量企业短期偿债能力，比率越高短期偿债能力越强。"}
        ]
    )
    ans1 = resp1.choices[0].message.content
    print("AI总结结果：")
    print(ans1)

    # 任务2：读取本地test.txt财务文档总结
    print("\n=====读取本地test.txt文档总结=====")
    with open("test.txt","r",encoding="utf-8") as f:
        file_text = f.read()

    resp2 = client.chat.completions.create(
        model="qwen3.8-flash",
        messages=[
            {"role":"system","content":"你是文本总结助手，简短总结下面文档内容"},
            {"role":"user","content": file_text}
        ]
    )
    ans2 = resp2.choices[0].message.content
    print("文档总结：")
    print(ans2)

except Exception as e:
    print(f"发生异常：{type(e)} {str(e)}")
    import traceback
    traceback.print_exc()