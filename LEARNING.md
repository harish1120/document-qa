# Learning Serverless + AWS SAM

A practical guide to getting comfortable with the deployment stack used in this project.

---

## The Mental Model First

Before memorizing commands, understand the three layers:

```text
Your code  →  Docker image  →  Infrastructure-as-code  →  AWS
(Python)      (Dockerfile)     (template.yaml / SAM)       (actual cloud)
```

SAM is just CloudFormation with shortcuts for Lambda. When you run `sam deploy`, it:

1. Turns your `template.yaml` into a CloudFormation stack
2. CloudFormation figures out what to create/update/delete
3. AWS provisions the actual resources

Everything in `template.yaml` maps 1:1 to something real in AWS. That's worth internalizing early.

---

## Phase 1 — Understand What You Already Have (1–2 days)

Don't touch new tutorials yet. Explore your own template first.

**Exercise 1:** Open the AWS Console after deploying. Find each resource SAM created:

- Lambda → Functions → your function → look at the config (memory, timeout, env vars)
- API Gateway → HTTP APIs → your API → look at routes
- S3 → your bucket → look at the folder structure after uploading a PDF
- CloudWatch → Log groups → `/aws/lambda/<function-name>` → read the logs from a real request

**Exercise 2:** Break something intentionally. Change `MemorySize: 1024` to `MemorySize: 128` in `template.yaml`, redeploy, and try indexing a large PDF. Watch it fail in CloudWatch logs. Then fix it.

**Exercise 3:** Read the [SAM resource reference](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-specification-resources-and-properties.html) for just the resources you're using: `AWS::Serverless::Function`, `AWS::Serverless::HttpApi`, `AWS::S3::Bucket`. Ignore everything else for now.

---

## Phase 2 — Build Something Smaller from Scratch (1 week)

The fastest way to understand SAM is to write a template with no existing code to lean on.

### Project: A simple URL shortener

- `POST /shorten` → saves `{short_code: url}` to DynamoDB → returns short URL
- `GET /{code}` → looks up code → redirects

This forces you to learn:

- `AWS::Serverless::SimpleTable` (DynamoDB shortcut)
- IAM permissions (your Lambda needs `dynamodb:PutItem`, `dynamodb:GetItem`)
- Path parameters in API Gateway (`/{code+}`)
- Lambda environment variables

Start here: [SAM Getting Started](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started-hello-world.html) — but only use it to learn `sam init` / `sam local invoke` / `sam deploy`. Write your own template logic.

---

## Phase 3 — Learn the Things That Will Actually Bite You

These aren't in most tutorials but you'll hit all of them:

### Cold starts

Lambda containers are recycled when idle. The first request after idle can take 2–5s. For this app, loading the FAISS index is expensive — that's why `rag.py` uses the ETag cache to avoid reloading on every warm invocation.

Read: [Understanding Lambda cold starts](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/)

### IAM permissions

Every AWS service call requires explicit permission. The most common error you'll see: `AccessDenied`. When it happens, read the error message — it always tells you exactly which action was denied and on which resource. Then add that action to your IAM role in `template.yaml`.

Practice: Remove `s3:HeadObject` from `BackendExecutionRole` in your template, deploy, try to ask a question, read the CloudWatch error.

### Lambda container image size

Your Lambda image pulls `langchain`, `faiss-cpu`, `torch` (indirectly), etc. Large images = slow cold starts and slow ECR pushes. Use `docker image inspect rag-backend-lambda` to check size. Slim it down by removing unused dependencies.

### CloudFormation change sets

Before `sam deploy` applies changes, it creates a "change set" — a diff of what will be created/modified/deleted. `sam deploy --guided` asks you to confirm it. Learn to read it. Deleting a resource (like an S3 bucket with data) will fail if you're not careful.

---

## Phase 4 — Extend This App

Real learning comes from adding features to something that already works:

| Feature | What you'd learn |
| --- | --- |
| Add a `DELETE /document` endpoint that removes a PDF from S3 and re-indexes | S3 delete operations, FAISS index management |
| Add a CloudFront distribution in front of API Gateway | CDN, cache headers, SAM for non-serverless resources |
| Store chat history in DynamoDB | `AWS::Serverless::SimpleTable`, session IDs |
| Add an SQS queue so indexing is async | Event-driven architecture, Lambda triggers |
| Set up a CI/CD pipeline with GitHub Actions | `sam deploy` in CI, ECR login in GitHub secrets |

The SQS one is particularly worth doing — right now, indexing blocks the HTTP request for 30+ seconds. A queue would let the upload return immediately, then index in the background.

---

## Resources Worth Your Time

| Resource | Why |
| --- | --- |
| [AWS SAM docs](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/) | Reference, not tutorial — ctrl+f what you need |
| [AWS Well-Architected Framework — Serverless Lens](https://docs.aws.amazon.com/wellarchitected/latest/serverless-applications-lens/welcome.html) | How AWS thinks about serverless design decisions |
| Yan Cui's blog [theburningmonk.com](https://theburningmonk.com) | Best practical Lambda/serverless writing; not beginner-padded |
| [CloudFormation resource types](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html) | When SAM shortcuts aren't enough, you need raw CloudFormation |

---

## The Most Important Habit

After every deploy, go to the CloudWatch logs and read them. Even when things work. You'll understand what Lambda is actually doing — init time, invocation time, memory used, errors. The console is your feedback loop.

```bash
# Tail logs from the CLI instead of clicking around:
sam logs -n BackendFunction --stack-name rag-app --tail
```
