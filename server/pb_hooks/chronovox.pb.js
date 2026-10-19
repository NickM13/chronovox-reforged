/// <reference path="../pb_data/types.d.ts" />
// After a version is uploaded, remove the oldest versions of that model beyond the limit in the
// "workspace_settings" collection (max_versions, 25 unless the owner changed it in the dashboard).
// Version numbers keep counting up; only the oldest copies are removed. A problem here never stops an upload.

onRecordAfterCreateSuccess((e) => {
  try {
    let keep = 25
    try {
      keep = e.app.findRecordById("workspace_settings", "chronovoxconfig").getInt("max_versions") || 25
    } catch (_) {}
    keep = Math.max(keep, 2)

    const all = e.app.findRecordsByFilter("versions", "model = {:model}", "-number", 0, 0, { model: e.record.get("model") })
    for (let i = keep; i < all.length; i++) {
      e.app.delete(all[i])
    }
  } catch (err) {
    e.app.logger().error("Chronovox: could not remove old versions: " + err)
  }
  e.next()
}, "versions")
