import { readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { isMap, isScalar, isSeq, parseDocument } from 'yaml'

const DEFAULT_BLUEPRINTS = ['render.yaml']
const BASE_RESOURCE_NAMES = ['digigrad', 'gradphone', 'gradphone-data']

const currentFile = fileURLToPath(import.meta.url)
const defaultRoot = resolve(dirname(currentFile), '..')

function getStringAt(map, key) {
  const value = map.get(key, true)
  return isScalar(value) && typeof value.value === 'string' ? value.value : null
}

function setStringAt(map, key, value) {
  const node = map.get(key, true)
  if (isScalar(node)) {
    node.value = value
    return
  }

  map.set(key, value)
}

function normalizeNamespace(value) {
  const namespace = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  if (!namespace) {
    throw new Error('Namespace must contain at least one letter or number.')
  }

  return namespace
}

function namespaceName(name, namespace) {
  if (name.startsWith(`${namespace}-`)) {
    return name
  }

  const baseName = BASE_RESOURCE_NAMES.find(
    (candidate) => name === candidate || name.endsWith(`-${candidate}`),
  )

  return `${namespace}-${baseName ?? name}`
}

function collectNameChange(map, namespace, nameChanges, changedNames) {
  const currentName = getStringAt(map, 'name')
  if (!currentName) {
    return
  }

  const nextName = namespaceName(currentName, namespace)
  if (nextName !== currentName) {
    nameChanges.set(currentName, nextName)
    changedNames.push({ from: currentName, to: nextName })

    for (const baseName of BASE_RESOURCE_NAMES) {
      if (currentName === baseName || currentName.endsWith(`-${baseName}`)) {
        nameChanges.set(baseName, nextName)
        break
      }
    }
    setStringAt(map, 'name', nextName)
  }
}

function collectServiceResources(serviceMap, namespace, nameChanges, changedNames) {
  collectNameChange(serviceMap, namespace, nameChanges, changedNames)

  const disk = serviceMap.get('disk', true)
  if (isMap(disk)) {
    collectNameChange(disk, namespace, nameChanges, changedNames)
  }
}

function updateReference(map, key, nameChanges) {
  const reference = map.get(key, true)
  if (!isMap(reference)) {
    return
  }

  const currentName = getStringAt(reference, 'name')
  if (!currentName) {
    return
  }

  const nextName = nameChanges.get(currentName)
  if (nextName) {
    setStringAt(reference, 'name', nextName)
  }
}

function getSeqAt(map, key) {
  const value = map.get(key, true)
  return isSeq(value) ? value : null
}

function collectNamesFromServiceSeq(seq, namespace, nameChanges, changedNames) {
  for (const item of seq.items) {
    if (isMap(item)) {
      collectServiceResources(item, namespace, nameChanges, changedNames)
    }
  }
}

function collectNamesFromSeq(seq, namespace, nameChanges, changedNames) {
  for (const item of seq.items) {
    if (isMap(item)) {
      collectNameChange(item, namespace, nameChanges, changedNames)
    }
  }
}

function visitMaps(node, visitor) {
  if (isMap(node)) {
    visitor(node)

    for (const item of node.items) {
      visitMaps(item.value, visitor)
    }
    return
  }

  if (isSeq(node)) {
    for (const item of node.items) {
      visitMaps(item, visitor)
    }
  }
}

function namespaceBlueprint(source, namespace) {
  const doc = parseDocument(source)
  const nameChanges = new Map()
  const changedNames = []

  if (isMap(doc.contents)) {
    const rootDatabases = getSeqAt(doc.contents, 'databases')
    const rootServices = getSeqAt(doc.contents, 'services')
    const projects = getSeqAt(doc.contents, 'projects')

    if (rootDatabases) {
      collectNamesFromSeq(rootDatabases, namespace, nameChanges, changedNames)
    }
    if (rootServices) {
      collectNamesFromServiceSeq(rootServices, namespace, nameChanges, changedNames)
    }

    if (projects) {
      for (const project of projects.items) {
        if (!isMap(project)) {
          continue
        }

        collectNameChange(project, namespace, nameChanges, changedNames)

        const environments = getSeqAt(project, 'environments')
        if (!environments) {
          continue
        }

        for (const environment of environments.items) {
          if (!isMap(environment)) {
            continue
          }

          const databases = getSeqAt(environment, 'databases')
          const services = getSeqAt(environment, 'services')

          if (databases) {
            collectNamesFromSeq(databases, namespace, nameChanges, changedNames)
          }
          if (services) {
            collectNamesFromServiceSeq(services, namespace, nameChanges, changedNames)
          }
        }
      }
    }
  }

  visitMaps(doc.contents, (map) => {
    updateReference(map, 'fromDatabase', nameChanges)
    updateReference(map, 'fromService', nameChanges)
  })

  return {
    contents: doc.toString(),
    changedNames,
  }
}

function parseArgs(argv) {
  const args = {
    blueprints: DEFAULT_BLUEPRINTS,
    namespace: process.env.GITHUB_ACTOR,
    root: defaultRoot,
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    const value = argv[index + 1]

    if (arg.startsWith('--namespace=')) {
      args.namespace = arg.slice('--namespace='.length)
      continue
    }

    if (arg.startsWith('--')) {
      switch (arg) {
        case '--namespace':
          if (!value) {
            throw new Error('Missing value for --namespace.')
          }
          args.namespace = value
          index += 1
          break
        case '--root':
          if (!value) {
            throw new Error('Missing value for --root.')
          }
          args.root = resolve(value)
          index += 1
          break
        default:
          throw new Error(`Unknown argument: ${arg}`)
      }
      continue
    }

    if (args.namespace && args.namespace !== process.env.GITHUB_ACTOR) {
      throw new Error(`Unexpected extra argument: ${arg}`)
    }

    args.namespace = arg
  }

  if (!args.namespace) {
    throw new Error(
      'Missing namespace. Run `npm run setup -- your-github-username`.',
    )
  }

  return {
    blueprints: args.blueprints,
    namespace: normalizeNamespace(args.namespace),
    root: args.root,
  }
}

async function setupAttendee({ blueprints, namespace, root }) {
  const changes = []

  for (const relativePath of blueprints) {
    const path = resolve(root, relativePath)
    const source = await readFile(path, 'utf8')
    const result = namespaceBlueprint(source, namespace)

    await writeFile(path, result.contents)
    changes.push({ path: relativePath, changedNames: result.changedNames })
  }

  return changes
}

function printSummary(changes) {
  for (const change of changes) {
    console.log(change.path)

    if (change.changedNames.length === 0) {
      console.log('  no changes')
      continue
    }

    for (const { from, to } of change.changedNames) {
      console.log(`  ${from} -> ${to}`)
    }
  }
}

async function main() {
  try {
    const args = parseArgs(process.argv.slice(2))
    const changes = await setupAttendee(args)
    printSummary(changes)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(message)
    process.exitCode = 1
  }
}

if (process.argv[1] === currentFile) {
  await main()
}

export {
  namespaceBlueprint,
  namespaceName,
  normalizeNamespace,
  parseArgs,
  setupAttendee,
}
