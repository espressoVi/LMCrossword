import json
import re
import os

def Filter(predictions):
    res = {}
    for key, pred in predictions.items():
        ans = pred.strip().lower()
        ans = ans.replace("clue: ","")
        ans = ans.replace("answer: ","")
        ans = re.sub(r'<(.*)>', "\n", ans)
        res[key] = ans
    return res

def postprocess_normal(predictions):
    res = Filter(predictions)
    for key, ans in res.items():
        ans = [i for i in ans.split("\n") if i]
        ans = "" if not ans else ans[0]
        ans = "".join([i for i in ans if i.isalpha()])
        res[key] = ans
    return res

def postprocess_counting(predictions):
    isnum = lambda f: all([i.isalnum() and not i.isalpha() for i in list(f)])
    res = Filter(predictions)
    for key, ans in res.items():
        ans = [i for i in ans.split("\n") if i and isnum(i)]
        ans = "" if not ans else ans[0]
        try:
            res[key] = int(ans)
        except:
            res[key] = -1
    return res

def postprocess_cot(predictions):
    res = {}
    for key, pred in predictions.items():
        ans = pred.strip().split()
        ans = [i for i in ans if i.isupper()]
        ans = "" if not ans else ans[-1]
        ans = ans.lower()
        ans = "".join([i for i in ans if i.isalpha()])
        res[key] = ans
    return res

def main():
    directory = "./outputs/"
    for file in os.listdir(directory):
        if ( 
                "json" not in file 
                or "jsonl" in file
                or "processed_" in file

        ):
            continue
        pred_file = os.path.join(directory, file)
        with open(pred_file, "r") as f:
            pred = json.load(f)
        if "CoT" in pred_file:
            res = postprocess_cot(pred)
        elif "counting" in pred_file:
            res = postprocess_counting(pred)
        else:
            res = postprocess_normal(pred)
        output_file = os.path.join(directory, f"processed_{file}")
        with open(output_file, "w") as f:
            json.dump(res, f, indent = 4)
    
if __name__ == "__main__":
    main()
