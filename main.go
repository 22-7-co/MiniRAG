package main

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/cloudwego/eino-ext/components/embedding/ark"
	inmilvus "github.com/cloudwego/eino-ext/components/indexer/milvus"
	"github.com/cloudwego/eino-ext/components/retriever/milvus"
	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
)

var ctx context.Context
var embedder *ark.Embedder
var indexer *inmilvus.Indexer
var spliter document.Transformer

func main() {
	err := godotenv.Load(".env")
	if err != nil {
		panic(err)
	}

	ctx = context.Background()
	InitClient()
	embedder = NewArkEmbedder(ctx)
	indexer = NewArkIndexer(ctx, embedder)
	retriever := NewArkRetriever(ctx, embedder)
	spliter = NewTrans(ctx)

	// insert()
	inquiry(retriever)

}

func insert() {
	content, err := os.OpenFile("./document.md", os.O_CREATE|os.O_RDWR, 0755) // create rdwr
	if err != nil {
		panic(err)
	}
	defer content.Close()
	bs, err := os.ReadFile("./document.md")
	if err != nil {
		panic(err)
	}
	docs := []*schema.Document{
		{
			ID:      "doc1",
			Content: string(bs),
		},
	}
	results, err := spliter.Transform(ctx, docs)
	if err != nil {
		panic(err)
	}
	for i, doc := range results {
		doc.ID = docs[0].ID + "_" + strconv.Itoa(i)
		println(doc.ID)
	}

	ids, err := indexer.Store(ctx, results)
	if err != nil {
		panic(err)
	}
	fmt.Println("存入完成，ids: ", ids)
}

func inquiry(retriever *milvus.Retriever) {
	// 查询
	text := "原生调用 LLM（如 OpenAI 的 API）只能做到："
	results, err := retriever.Retrieve(ctx, text)
	if err != nil {
		panic(err)
	}
	println("查询结果：")
	for _, doc := range results {
		println(doc.ID)
		println(doc.Content)
		println("--------------")
	}
}
