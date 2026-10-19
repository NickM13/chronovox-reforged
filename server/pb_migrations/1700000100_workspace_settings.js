/// <reference path="../pb_data/types.d.ts" />
// Workspace limits the owner can see and change in the dashboard, and the size limits for uploads.
//
// collection "workspace_settings" (one record), field max_versions: how many versions of each shared model are kept.
//   Older ones are removed when a new one is uploaded (see pb_hooks/chronovox.pb.js). The editor reads this number
//   to tell people about it.
// versions.file: the largest model file accepted is 4 MB (most models are a few KB), and the preview picture next to
//   it 512 KB (a real one is about 30 KB). The editor knows the same 4 MB (MAX_FILE_BYTES in
//   chronovox/workspace_api.py). This only affects new uploads.

migrate((app) => {
  const settings = new Collection({
    type: "base",
    name: "workspace_settings",
    listRule: '@request.auth.id != ""',
    viewRule: '@request.auth.id != ""',
    createRule: null,
    updateRule: null,
    deleteRule: null,
    fields: [
      { name: "max_versions", type: "number", required: true, min: 2, max: 1000, onlyInt: true },
    ],
  })
  app.save(settings)

  const record = new Record(settings)
  record.id = "chronovoxconfig"
  record.set("max_versions", 25)
  app.save(record)

  const versions = app.findCollectionByNameOrId("versions")
  versions.fields.getByName("file").maxSize = 4194304       // 4 MB
  versions.fields.getByName("preview").maxSize = 524288     // 512 KB
  app.save(versions)
}, (app) => {
  try { app.delete(app.findCollectionByNameOrId("workspace_settings")) } catch (_) {}
  const versions = app.findCollectionByNameOrId("versions")
  versions.fields.getByName("file").maxSize = 268435456     // back to 256 MB
  versions.fields.getByName("preview").maxSize = 5242880    // back to 5 MB
  app.save(versions)
})
