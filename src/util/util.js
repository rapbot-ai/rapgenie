const { existsSync, readFileSync, readdir, rmdir, unlink, } = require('fs');
const path = require('path');
const { promisify } = require('util');
const { createHash } = require("crypto");

const readdirProm = promisify(readdir);
const rmdirProm = promisify(rmdir);
const unlinkProm = promisify(unlink);

const rmDirsRecursively = async (dir) => {
  const entries = await readdirProm(dir, { withFileTypes: true });
  await Promise.all(entries.map(entry => {
    const fullPath = path.join(dir, entry.name);
    return entry.isDirectory() ? rmDirsRecursively(fullPath) : unlinkProm(fullPath);
  }));
  await rmdirProm(dir);
  return true
};

const getFileChecksum = (songPath) => {
  if (existsSync(songPath)) {
    const hash = createHash("sha1");
    hash.update(readFileSync(songPath));
    return hash.digest("hex");
  }
}

module.exports = {
  rmDirsRecursively,
  getFileChecksum,
}