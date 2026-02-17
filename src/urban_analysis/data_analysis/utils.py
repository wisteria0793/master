import re

def extract_district_from_address(address: str) -> str:
    """
    住所文字列から「函館市」の後の町名を抽出する。
    """
    if not isinstance(address, str): return None
    match = re.search(r'函館市([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+?)(?:[0-9\-－‐‑‒–—―−⸻﹣－]|丁目|番地|町|村|大字|字|$)', address)
    if match:
        district = match.group(1).strip()
        if '函館山' in district: return '函館山'
        return district
    return None

def format_hakodate_address(address: str) -> str:
    """
    住所文字列が「函館市」で始まるように整形する。
    """
    if not isinstance(address, str):
        return address
    hako_index = address.find("函館市")
    if hako_index != -1:
        return address[hako_index:]
    else:
        return f"函館市{address}"
