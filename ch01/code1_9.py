from langchain_core.output_parsers import CommaSeparatedListOutputParser
# 랭체인은 CSV, XML 등 다양한 경우에 대응하는 출력 파서를 제공한다.

parser = CommaSeparatedListOutputParser()
items = parser.parse('apple, banana, cherry')

print(items)