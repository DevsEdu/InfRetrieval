import Tools

from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.fields import *

from whoosh.qparser import QueryParser
from whoosh import scoring

def indexing(paths):   

    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
    ix = create_in("indexdir", schema)
    writer = ix.writer()

    for path in paths:
        writer.add_document(title=path, content=Tools.get_document(path))

    writer.commit()

def get_query():
    query = input("O que deseja buscar? ")

    ix = open_dir("indexdir")

    bm25f = scoring.BM25F()
    with ix.searcher(weighting=bm25f) as searcher:
        query = QueryParser("content", ix.schema).parse(query)
        results = searcher.search(query)

        for r in results:
            print(f'{r["title"]}\t -----\t {r.score}')

def main():
    paths = Tools.get_paths()

    #indexing(paths)

    get_query()

if __name__ == "__main__":
    main()