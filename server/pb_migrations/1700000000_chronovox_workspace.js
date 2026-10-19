/// <reference path="../pb_data/types.d.ts" />
// Sets up the Chronovox Reforged workspace: invite codes, shared models and their versions.
// PocketBase runs this file by itself the first time it starts with this folder next to it.

migrate((app) => {
  // Invite codes: a person can only register with a code that exists here. Add and remove them in the dashboard.
  const invites = new Collection({
    type: "base",
    name: "invites",
    listRule: null,
    viewRule: null,
    createRule: null,
    updateRule: null,
    deleteRule: null,
    fields: [
      { name: "code", type: "text", required: true, min: 6, max: 64 },
      { name: "note", type: "text", max: 200 },
    ],
    indexes: ["CREATE UNIQUE INDEX idx_invites_code ON invites (code)"],
  })
  app.save(invites)

  // Registration needs a valid invite code, and people can only see and change their own account.
  const users = app.findCollectionByNameOrId("users")
  users.createRule = '@request.body.invite != "" && @collection.invites.code ?= @request.body.invite'
  users.listRule = '@request.auth.id != ""'
  users.viewRule = '@request.auth.id != ""'
  users.updateRule = "id = @request.auth.id"
  users.deleteRule = "id = @request.auth.id"
  app.save(users)

  // A shared model: a name and who first shared it. Its contents are the versions below.
  const models = new Collection({
    type: "base",
    name: "models",
    listRule: '@request.auth.id != ""',
    viewRule: '@request.auth.id != ""',
    createRule: '@request.auth.id != "" && @request.body.owner = @request.auth.id',
    updateRule: "owner = @request.auth.id",
    deleteRule: "owner = @request.auth.id",
    fields: [
      { name: "name", type: "text", required: true, min: 1, max: 120 },
      { name: "owner", type: "relation", required: true, collectionId: users.id, maxSelect: 1 },
      { name: "created", type: "autodate", onCreate: true, onUpdate: false },
      { name: "updated", type: "autodate", onCreate: true, onUpdate: true },
    ],
  })
  app.save(models)

  // One saved copy of a shared model. Versions are never changed once uploaded. Two people uploading the same
  // number at the same moment cannot both succeed (the unique index), which is how "someone else got there first"
  // is detected.
  const versions = new Collection({
    type: "base",
    name: "versions",
    listRule: '@request.auth.id != ""',
    viewRule: '@request.auth.id != ""',
    createRule: '@request.auth.id != "" && @request.body.author = @request.auth.id',
    updateRule: null,
    deleteRule: null,
    fields: [
      { name: "model", type: "relation", required: true, collectionId: models.id, maxSelect: 1, cascadeDelete: true },
      { name: "number", type: "number", required: true, min: 1, onlyInt: true },
      { name: "author", type: "relation", required: true, collectionId: users.id, maxSelect: 1 },
      { name: "note", type: "text", max: 500 },
      { name: "sha256", type: "text", max: 64 },
      { name: "file", type: "file", required: true, maxSelect: 1, maxSize: 268435456, protected: true },
      { name: "preview", type: "file", maxSelect: 1, maxSize: 5242880, protected: true,
        mimeTypes: ["image/png"] },
      { name: "created", type: "autodate", onCreate: true, onUpdate: false },
    ],
    indexes: ["CREATE UNIQUE INDEX idx_versions_model_number ON versions (model, number)"],
  })
  app.save(versions)
}, (app) => {
  for (const name of ["versions", "models", "invites"]) {
    try { app.delete(app.findCollectionByNameOrId(name)) } catch (_) {}
  }
})
