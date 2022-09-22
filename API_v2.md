# API docs for Flowspeech v2

The source of truth for API endpoints (requests and responses) is swagger.
https://sp.flowchase.app/swagger-ui/

Requests and responses should be JSON.
The audio is base64 encoded files from these format: ogg, caf. It can be others as long as (it is handled by libsndfile)[http://www.mega-nerd.com/libsndfile/]

There could still be responses in HTML for unexpected errors (e.g. from nginx, a 502 bad gateway)

For 400 and 500 errors, it will be JSON with only `error` and `status` payloads

Otherwise, there is a `detected` flag and other payloads as described in swagger.
```typescript
type Response = {
    error: bool
    status: string| {"type": string, "value": string,   "traceback":string}
    detected: 'speech'|'nospeech'|'nonsense'|'empty_audio'
 ...
}
```