# API docs for Flowspeech v2

The source of truth for API endpoints (requests and responses) is swagger.
http://135.125.247.39/swagger-ui/

Requests and responses should be JSON.
There could still be responses in HTML for unexpected errors (e.g. from nginx, a 502 bad gateway)

For 400 and 500 errors, it will be JSON with only `error` and `status` payloads

Otherwise, there is a `detected` flag and other payloads as described in swagger.
```typescript
type Response = {
    error: bool
    status: string
    detected: 'speech'|'nospeech'|'nonsense'
 ...
}
```