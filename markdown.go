package main

import (
	"context"

	"github.com/cloudwego/eino-ext/components/document/transformer/splitter/markdown"
	"github.com/cloudwego/eino/components/document"
)

func NewTrans(ctx context.Context) document.Transformer{
	spliter, err := markdown.NewHeaderSplitter(ctx, &markdown.HeaderConfig{
		Headers: map[string]string{
			"#":"h1",
			"##":"h2",
			"###":"h3",
		},
		TrimHeaders: false,
	})
	if err != nil {
		panic(err)
	}
	return spliter
}