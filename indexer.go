package main

import (
	"context"

	"github.com/cloudwego/eino-ext/components/embedding/ark"
	"github.com/cloudwego/eino-ext/components/indexer/milvus"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func NewArkIndexer(ctx context.Context, embedder *ark.Embedder) *milvus.Indexer {
	var collection = "AwesomeEino"

	var fields = []*entity.Field{
		{
			Name:     "id",
			DataType: entity.FieldTypeVarChar,
			TypeParams: map[string]string{
				"max_length": "255",
			},
			PrimaryKey: true,
		},
		{
			Name:     "vector", // 确保字段名匹配
			DataType: entity.FieldTypeBinaryVector,
			TypeParams: map[string]string{
				"dim": "81920",
			},
		},
		{
			Name:     "content",
			DataType: entity.FieldTypeVarChar,
			TypeParams: map[string]string{
				"max_length": "8192",
			},
		},
		{
			Name:     "metadata",
			DataType: entity.FieldTypeJSON,
		},
	}
	indexer, err := milvus.NewIndexer(ctx, &milvus.IndexerConfig{
		Client:MilvusCli,
		Collection:collection,
		Fields: fields,
		Embedding: embedder,
	})
	if err != nil {
		panic(err)
	}
	return indexer
}