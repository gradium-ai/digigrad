import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { test } from 'node:test'

const execFileAsync = promisify(execFile)
const repoRoot = new URL('..', import.meta.url).pathname
const setupScript = join(repoRoot, 'scripts/setup-attendee.js')

const blueprintFixture = `previews:
  generation: off

projects:
  - name: digigrad
    environments:
      - name: production
        services:
          - type: web
            name: gradphone
            disk:
              name: gradphone-data
              mountPath: /data
              sizeGB: 1
`

async function copyBlueprintToTempRepo() {
  const root = await mkdtemp(join(tmpdir(), 'digigrad-setup-'))
  await writeFile(join(root, 'render.yaml'), blueprintFixture)
  return root
}

async function runSetup(root, namespace = 'Octo.User') {
  return execFileAsync(process.execPath, [
    setupScript,
    '--root',
    root,
    '--namespace',
    namespace,
  ])
}

test('setup script namespaces project, service, and disk names', async () => {
  const root = await copyBlueprintToTempRepo()

  await runSetup(root)

  const blueprint = await readFile(join(root, 'render.yaml'), 'utf8')
  assert.match(blueprint, /name: octo-user-digigrad/)
  assert.match(blueprint, /name: production/)
  assert.match(blueprint, /name: octo-user-gradphone/)
  assert.match(blueprint, /name: octo-user-gradphone-data/)
})

test('setup script is idempotent for the same namespace', async () => {
  const root = await copyBlueprintToTempRepo()

  await runSetup(root)
  await runSetup(root)

  const blueprint = await readFile(join(root, 'render.yaml'), 'utf8')
  assert.doesNotMatch(blueprint, /octo-user-octo-user-/)
})

test('setup script uses GITHUB_ACTOR when namespace is omitted', async () => {
  const root = await copyBlueprintToTempRepo()

  await execFileAsync(process.execPath, [setupScript, '--root', root], {
    env: {
      ...process.env,
      GITHUB_ACTOR: 'Button.User',
    },
  })

  const blueprint = await readFile(join(root, 'render.yaml'), 'utf8')
  assert.match(blueprint, /name: button-user-digigrad/)
  assert.match(blueprint, /name: button-user-gradphone-data/)
})

test('setup script replaces a previous attendee prefix', async () => {
  const root = await copyBlueprintToTempRepo()

  await runSetup(root, 'First.User')
  await runSetup(root, 'Second.User')

  const blueprint = await readFile(join(root, 'render.yaml'), 'utf8')
  assert.match(blueprint, /name: second-user-digigrad/)
  assert.match(blueprint, /name: second-user-gradphone/)
  assert.doesNotMatch(blueprint, /second-user-first-user-/)
})
