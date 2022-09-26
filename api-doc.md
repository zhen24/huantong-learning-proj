# 数据接收服务

- 请求方式：POST
- 请求地址： http://<host>:8118/label/commit
- 请求包体数据类型：json 列表
- 请求包体示例：
    ```json
  {
      "label_id": "该条标注的ID",

      "tenant_id": "客户 Tenant ID",
      "request_at": "发送匹配请求的时间",
      "response_at": "响应匹配请求的时间",
      "query_info": {
          "query_name": "待匹配名称",
          "upstream_id": "上游机构ID",

          "upstream_name": "上游机构名称",
          "province": "机构所在省",
          "city": "机构所在市",
          "country": "机构所在县",
          "address": "机构地址",
          "status": "机构有效状态(28为失效,41为有效)",
          "alias_names": [
              {
                  "tenant_id": "客户 Tenant ID",
                  "organization_id": "机构ID",
                  "name": "别名1"
              },
              {
                  "tenant_id": "客户 Tenant ID",
                  "organization_id": "机构ID",
                  "name": "别名2"
              }
          ],

          "query_name_tokens": "待|匹配|名称"
      },

      "match_results": [
          {
              "downstream_id": "下游机构ID",

              "downstream_name": "下游机构名称",
              "province": "机构所在省",
              "city": "机构所在市",
              "country": "机构所在县",
              "address": "机构地址",
              "status": "机构有效状态(28为失效,41为有效)",
              "alias_names": [
                  {
                      "tenant_id": "客户 Tenant ID",
                      "organization_id": "机构ID",
                      "name": "别名1"
                  },
                  {
                      "tenant_id": "客户 Tenant ID",
                      "organization_id": "机构ID",
                      "name": "别名2"
                  }
              ]
          },
          {
              "downstream_id": "下游机构ID",
              "downstream_name": "下游机构名称",
              "province": "机构所在省",
              "city": "机构所在市",
              "country": "机构所在县",
              "address": "机构地址",
              "status": "机构有效状态(28为失效,41为有效)",
              "alias_names": [
                  {
                      "tenant_id": "客户 Tenant ID",
                      "organization_id": "机构ID",
                      "name": "别名1"
                  },
                  {
                      "tenant_id": "客户 Tenant ID",
                      "organization_id": "机构ID",
                      "name": "别名2"
                  }
              ]
          }
      ],

      "checked_result": {
          "downstream_id": "下游机构ID",

          "downstream_name": "下游机构名称",
          "province": "机构所在省",
          "city": "机构所在市",
          "country": "机构所在县",
          "address": "机构地址",
          "status": "机构有效状态(28为失效,41为有效)",
          "alias_names": [
              {
                  "tenant_id": "客户 Tenant ID",
                  "organization_id": "机构ID",
                  "name": "别名1"
              },
              {
                  "tenant_id": "客户 Tenant ID",
                  "organization_id": "机构ID",
                  "name": "别名2"
              }
          ]
      },

      "labeled_results": [
          {
              "downstream_id": "下游机构ID",

              "downstream_name": "下游机构名称",
              "province": "机构所在省",
              "city": "机构所在市",
              "country": "机构所在县",
              "address": "机构地址",
              "status": "机构有效状态(28为失效,41为有效)",
              "alias_names": [
                  {
                      "tenant_id": "客户 Tenant ID",
                      "organization_id": "机构ID",
                      "name": "别名1"
                  },
                  {
                      "tenant_id": "客户 Tenant ID",
                      "organization_id": "机构ID",
                      "name": "别名2"
                  }
              ],

              "modified_query_name_tokens": "待|匹配|名称"
          }
      ]
  }
    ```

- 参数说明

参数                 |   类型    | 释义
--------------------|-----------|------
label_id            | integer | 该条标注的ID
tenant_id           | integer | 客户 Tenant ID
request_at          | string  | 发送匹配请求的时间
response_at         | string  | 响应匹配请求的时间
query_info          | json    | 请求附带的信息
match_results       | array   | 匹配结果附带的信息
checked_result      | json    | 人工复核结果的信息
labeled_results     | array   | 人工标注结果的信息

---

- 返回结果:
    - 200: No Content --入库成功, 返回200
    - 400: Bad Request --入库失败, 返回400
      ```json
      {
          "errorMessage": "缺失标注结果"
      }
      ```

# 训练任务
注: 一经开始, 不可中断

- 请求方式：POST
- 请求地址： http://<host>:8118/train/commit
- 请求包体数据类型：json 列表
- 请求包体示例：

  ```json
    {
        "task_id": 111111,
        "name": "第一次优化",
        "description": "第一次优化"
    }
  ```

- 参数说明

参数         |   类型     | 释义
------------|-----------|------------
task_id     | integer   | 训练任务ID
name        | string    | 任务名称，具有唯一性
description | string    | 任务描述

---

- 返回结果:
    - 200: The contents are as follows --入库成功, 返回200
        - 其任务状态status分为: 准备中、训练中、训练失败、训练完成、使用中
      ```json
      {
          "task_id": 111111,
          "name": "第一次优化",
          "description": "第一次优化",
          "status": "准备中",
          "storage_path": "",
          "create_at": "2022-09-26 18:21:30",
          "finished_at": null,
          "deleted_at": null
      }
      ```
    - 400: Bad Request --入库失败, 返回400
      ```json
      {
          "errorMessage": "已存在正在执行的训练任务!!!"
      }
      ```

# 更新任务
注: 一经开始, 不可中断

- 请求方式：POST
- 请求地址： http://<host>:8118/model/commit
- 请求包体数据类型：json 列表
- 请求包体示例：

  ```json
    {
        "task_id": 111111,
        "name": "第一次更新",
        "description": "第一次更新",
        "model_source": "第一次优化"
    }
  ```

- 参数说明

参数          |   类型     | 释义
-------------|-----------|------------
task_id      | integer   | 更新任务ID
name         | string    | 任务名称
description  | string    | 任务描述
model_source | string    | 模型出处,通过选择模型可得

---

- 返回结果:
    - 200: The contents are as follows --入库成功, 返回200
        - 其任务状态status分为: 升级中、升级失败、升级完成
      ```json
      {
          "task_id": 111111,
          "name": "第一次优化",
          "description": "第一次优化",
          "model_source": "选取的训练任务名称",
          "status": "升级中",
          "create_at": "2022-09-26 18:21:30",
          "finished_at": null,
          "deleted_at": null
      }
      ```
    - 400: Bad Request --入库失败, 返回400
      ```json
      {
          "errorMessage": "已存在正在执行的升级任务!!!"
      }
      ```
